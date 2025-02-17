Let the Agent Decide: LLM-Driven Onchain Trading System


Overview


Welcome to the repository for our project on leveraging Large Language Models (LLMs) for real-time onchain trading decisions. This system utilizes LLMs to analyze onchain and offchain data, make portfolio allocation decisions, and execute trades autonomously. It aims to bridge the gap for blockchain users without automation frameworks by offering a personal agent for trading management.  The LLM is currently live on mainnet and is tasked with periodically rebalancing an Arbitrum wallet.  


Features Used:

 - Onchain Data: Wallet transactions, token movements, and smart contract interactions.

- Offchain Data: Real-time market prices, trading news, and sentiment analysis.

- Macro Data: Real-time interest rate and cpi data from OpenBB

- LLM-Driven Reasoning: Logical steps for trade proposals based on gathered data.

- Trade Execution: Automated buy, sell, modify, or hold actions via blockchain platforms.



Methodology


1. Data Collection:


- Onchain data fetched via tools like Uniswap API and Flipside Crypto SQL.

- Offchain data gathered from market APIs and news feeds like OpenBB


2. Analysis and Modeling:


- LLMs analyze inputs for reasoning and trade decisions.

- Python-based integration ensures smooth data preprocessing and model execution.


3. Execution Framework:


- Trades are settled on blockchain platforms (Uniswap by default).




Key Features and Findings


- Data Experimentation: Combined multiple data sources for trade proposal improvements.

- Reasoning Steps: Demonstrated clear, step-by-step trade logic.

- Impactful Trades: Focused on maximizing portfolio returns with minimal risk.

- Real-Time Execution: Optimized smaller models for decisions within 2 seconds.



Project Structure


- /data: Raw and processed data for model training and evaluation.

- /notebooks: Jupyter notebooks showcasing experiments and results.

- /templates: HTML/CSS/JS used for the front end.

- /python_scripts: Python helper functions.

- /sql_queries: Dynamic SQL queries we pass to the Flipside API.

- requirements.txt: Python dependencies for the project.

- trade_app.py: The Python script which contains the LLM trading logic; this re-runs every 7 days.

- data_app.py: The Python script which contains the LLM portfolio data; this re-runs daily.



Applications


- Automated Portfolio Management: Optimize asset allocation using LLM reasoning.

- Personalized Trading Agent: Tailor trade decisions to user preferences.

- Trade Evaluation Framework: Analyze trade proposals for transparency and quality.



Technologies


- LLM Integration: OpenAI GPT-4o.

- Blockchain Tools: Web3.py, Uniswap-Python, Flipside Crypto SQL.
  
- Market/News Tools: OpenBB.

- Front-End: HTML, CSS, JS.

- Back-End: Python, SQL.

- Deployment: Ngrok.



Contribution


We welcome contributions to improve this project! Fork the repository, make your changes, and submit a pull request. 

Suggestions and new ideas are greatly appreciated.

Thank you for exploring this project. Let’s redefine onchain trading together! 🚀

---

Flipside EDA: https://flipsidecrypto.xyz/studio/queries/111ce9b8-eaf9-4364-bae6-079c0b99c424

You will need to create a new virtual environment, run pip install -r requirements.txt, and to run the jupyter notebooks you need to create a Jupyter kernel.

https://medium.com/@WamiqRaza/how-to-create-virtual-environment-jupyter-kernel-python-6836b50f4bf4

Environment variables are accessed via .env file in root directory. You will need to create your own and fill the variables with your own keys.
