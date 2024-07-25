# LID: Token Classification

🇧🇷 [README](README.md) | 🇺🇸 [README](README.en.md)

Este repositório contém o código e os datasets utilizados na resolução do problema do processo seletivo da equipe de IA do projeto Lia e LEDES em 2024.2.

## Descrição do problema: Token Classification

Sua tarefa será extrair de textos coletados do InfoMoney as entidades de: empresa, empresario, politico, outras_pessoas, valor_financeiro, cidade, estado, pais, organização e banco.

### Parte 1: Coleta de dados

Façam scrapy do site [InfoMoney](https://www.infomoney.com.br/). Sugestão: vejam que o site possui um robots.txt.

### Parte 2: Rotulação

Instale localmente em sua máquina o [label-studio](https://labelstud.io/) e rotule as classes definidas no problema. Na interface do Label-Studio vejam o template de Natural Language Processing -> Named Entity Recognition.

### Parte 3: Treinamento

Treinar o model.

### Parte 4: Avaliação

Avaliar o modelo com precisão, recall e f1-score.

### Parte 5: Deploy

Colocar no Label-Studio o modelo para fazer pré-rotulação.

### Exemplos de classes

- empresa: Petrobrás, Carrefour.
- empresario: Abilio Diniz.
- politico: Lula, Bolsonaro.
- outras_pessoas: George Clooney, Julia Roberts.
- valor_financeiro: US$ 6,5 bilhões; R$ 2,0; US$ 0,38.
- cidade: Campo Grande, Nova York.
- estado: MS, Nova York.
- pais: Brasil, Japão.
- organização: Fundo das Nações Unidas para Alimentação e Agricultura (FAO);Organização Não Governamental Ação da Cidadania; Organização para a Cooperação e o Desenvolvimento Econômico (OCDE).
- banco: Banco Nacional de Desenvolvimento Econômico e Social (BNDES); Bradesco, Itaú.

### Entrega

Entrega via email com os links para um vídeo de 15 minutos e todos os código no GitHub.

No repositório do github deve conter um código driver com o nome reproduzir.py que ao chamá-lo deve conseguir treinar o modelo.

## Instruções para execução

O projeto foi executado em um ambiente Linux, portanto, as instruções a seguir são para esse sistema operacional.

### Reprodução do scrap e treinamento

1. Clone o repositório:

    ```bash
    git clone https://github.com/TiagoSanti/LID-token-classification.git
    ```

2. Acesse o diretório do projeto:

    ```bash
    cd LID-token-classification
    ```

3. Crie um ambiente virtual:

    ```bash
    python -m venv venv
    ```

4. Ative o ambiente virtual:

    ```bash
    source venv/bin/activate
    ```

5. Instale as dependências do projeto:

    ```bash
    pip install -r requirements_linux.txt
    ```

- Execute o script `scrap.py` para coletar os dados do site InfoMoney:

    ```bash
    python scrap.py
    ```

- Execute o script `reproduce.py` para treinar o modelo:

    ```bash
    python reproduce.py
    ```

    Opcional: crie um arquivo `.env` com as variáveis de ambiente `WANDB_API_KEY` e `HUGGINGFACE_API_KEY` para salvar os resultados do treinamento no [Wandb](https://wandb.ai/) e o modelo treinado no [HuggingFace](https://huggingface.co/).

    ```bash
    WANDB_API_KEY=your_wandb_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

### Reprodução do deploy

1. Clone o repositório:

    ```bash
    git clone https://github.com/TiagoSanti/LID-token-classification.git
    ```

2. Acesse o diretório `ml-backend` do projeto:

    ```bash
    cd LID-token-classification/ml-backend
    ```

3. Crie um ambiente virtual:

    ```bash
    python -m venv venv
    ```

4. Ative o ambiente virtual:

    ```bash
    source venv/bin/activate
    ```

5. Instale as dependências do projeto:

    ```bash
    pip install -r requirements.txt
    ```

6. Execute o comando para iniciar o servidor:

    ```bash
    label-studio-ml start .
    ```

7. Adicione o endereço `http://localhost:9090/` no Label-Studio para fazer a pré-rotulação.

## Wandb e HuggingFace

Os resultados dos treinamentos do modelo estão disponíveis no [Wandb](https://wandb.ai/tiagosanti/ner-finetuning/workspace?nw=nwusertiagosanti) e o modelo treinado está disponível no [HuggingFace](https://huggingface.co/TiagoSanti/bert-ner-finetuned/tree/main).
