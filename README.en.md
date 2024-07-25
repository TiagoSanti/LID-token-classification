# LID: Token Classification

🇧🇷 [README](README.md) | 🇺🇸 [README](README.en.md)

This repository contains the code and datasets used in solving the selective process problem for the AI team of the Lia and LEDES project in 2024.2.

## Problem Description: Token Classification

Your task will be to extract the following entities from texts collected from InfoMoney: empresario, politico, outras_pessoas, valor_financeiro, cidade, estado, pais, organização e banco.

### Part 1: Data Collection

Perform scraping from the [InfoMoney](https://www.infomoney.com.br/) website. Suggestion: note that the site has a robots.txt.

### Part 2: Labeling

Install [label-studio](https://labelstud.io/) locally on your machine and label the classes defined in the problem. In the Label-Studio interface, check the Natural Language Processing -> Named Entity Recognition template.

### Part 3: Training

Train the model.

### Part 4: Evaluation

Evaluate the model using precision, recall, and f1-score.

### Part 5: Deployment

Place the model in Label-Studio for pre-labeling.

### Examples of Classes

- empresa (company): Petrobras, Carrefour.
- empresario (entrepreneur): Abilio Diniz.
- politico (politician): Lula, Bolsonaro.
- outras_pessoas (other_people): George Clooney, Julia Roberts.
- valor_financeiro (financial_value): US$ 6.5 billion; R$ 2.0; US$ 0.38.
- cidade (city): Campo Grande, New York.
- estado (state): MS, New York.
- pais (country): Brazil, Japan.
- organização (organization): United Nations Food and Agriculture Organization (FAO); Non-Governmental Organization Ação da Cidadania; Organization for Economic Co-operation and Development (OECD).
- banco (bank): National Bank for Economic and Social Development (BNDES); Bradesco, Itaú.

### Submission

Submit via email with links to a 15-minute [video]() and all the code on GitHub.

The GitHub repository must contain a driver code named `reproduce.py` that, when called, should be able to train the model.

## Execution Instructions

The project was executed in a Linux environment, therefore, the following instructions are for this operating system.

### Reproduction of Scraping and Training

1. Clone the repository:

    ```bash
    git clone https://github.com/TiagoSanti/LID-token-classification.git
    ```

2. Access the project directory:

    ```bash
    cd LID-token-classification
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the project dependencies:

    ```bash
    pip install -r requirements_linux.txt
    ```

- Execute the `scrap.py` script to collect data from the InfoMoney website:

    ```bash
    python scrap.py
    ```

- Execute the `reproduce.py` script to train the model:

    ```bash
    python reproduce.py
    ```

    Optional: create a `.env` file with the environment variables `WANDB_API_KEY` and `HUGGINGFACE_API_KEY` to save the training results on [Wandb](https://wandb.ai/) and the trained model on [HuggingFace](https://huggingface.co/).

    ```bash
    WANDB_API_KEY=your_wandb_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

### Reproduction of Deployment

1. Clone the repository:

    ```bash
    git clone https://github.com/TiagoSanti/LID-token-classification.git
    ```

2. Access the `ml-backend` directory of the project:

    ```bash
    cd LID-token-classification/ml-backend
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. Execute the command to start the server:

    ```bash
    label-studio-ml start .
    ```

7. Add the address `http://localhost:9090/` in Label-Studio for pre-labeling.

## Wandb and HuggingFace

The training results of the model are available on [Wandb](https://wandb.ai/tiagosanti/ner-finetuning/workspace?nw=nwusertiagosanti) and the trained model is available on [HuggingFace](https://huggingface.co/TiagoSanti/bert-ner-finetuned/tree/main).

## Video

The video presenting the project is available on [YouTube]().
