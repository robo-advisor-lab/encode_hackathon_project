from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3, EthereumTesterProvider
from web3.exceptions import TransactionNotFound,TimeExhausted

def get_token_decimals(TOKEN_CONTRACTS,w3):
        
        """Fetch decimals for each token contract using Web3."""
        try:
            # ERC20 ABI for decimals function
            decimals_abi = [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                }
            ]

            # Dictionary to store decimals for each token
            token_decimals = {}

            for token, address in TOKEN_CONTRACTS.items():
                if address:
                    try:
                        contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=decimals_abi)
                        decimals = contract.functions.decimals().call()
                        token_decimals[token] = decimals
                    except Exception as e:
                        print(f"Error fetching decimals for {token}: {e}")
                        token_decimals[token] = None
                else:
                    print(f"Contract address for {token} is not set.")
                    token_decimals[token] = None

            return token_decimals

        except Exception as e:
            print(f"Error: {e}")
            return None

def get_balance(TOKEN_CONTRACTS, TOKEN_DECIMALS, ACCOUNT_ADDRESS,w3):
    """Fetch token balances using Web3 with provided decimal adjustments."""
    try:
        # ERC20 ABI for balanceOf function
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }
        ]

        # Fetch balances using the provided decimals
        balances = {}
        for token, address in TOKEN_CONTRACTS.items():
            decimals = TOKEN_DECIMALS.get(token)

            if address and decimals is not None:
                try:
                    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=erc20_abi)
                    balance_wei = contract.functions.balanceOf(ACCOUNT_ADDRESS).call()
                    balances[token] = balance_wei / 10**decimals
                except Exception as e:
                    print(f"Error fetching balance for {token}: {e}")
                    balances[token] = None
            else:
                print(f"Skipping {token} due to missing address or decimals.")
                balances[token] = None

        # Print and return balances
        print(f"Balances for account {ACCOUNT_ADDRESS}: {balances}")
        return balances

    except Exception as e:
        print(f"Error fetching balances: {e}")
        return None

def convert_to_usd(balances, prices,TOKEN_CONTRACTS):
    """
    Convert token balances to their USD equivalent using token prices.

    Parameters:
    - balances (dict): Dictionary of token balances.
    - prices (dict): Dictionary of token prices.

    Returns:
    - dict: Dictionary of token balances converted to USD.
    """
    # Convert token keys to upper case for consistency
    print(f'balances: {balances.keys()}')
    print(f'TOKEN_CONTRACTS.keys(): {TOKEN_CONTRACTS.keys()}')

    for token in TOKEN_CONTRACTS.keys():
        if f"{token}" not in prices:
            print(f"Missing price for token: {token}")

    usd_balances = {
        token: balances[token] * prices[f"{token}"]
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}" in prices
    }
    return usd_balances