import os
from dotenv import load_dotenv
import pandas as pd
import sqlite3
import google.generativeai as genai

# -------------------- Setup Phase --------------------

# Load data
cabecalho_path = os.path.join('data', '202401_NFs_Cabecalho.csv')
itens_path = os.path.join('data', '202401_NFs_Itens.csv')

df_cabecalho = pd.read_csv(cabecalho_path, sep=',',
                           decimal='.', encoding='utf-8')
df_itens = pd.read_csv(itens_path, sep=',', decimal='.', encoding='utf-8')

# Normalize column names
df_cabecalho.columns = df_cabecalho.columns.str.strip(
).str.upper().str.replace(" ", "_")
df_itens.columns = df_itens.columns.str.strip().str.upper().str.replace(" ", "_")

# Standardize key name
df_cabecalho.rename(
    columns={df_cabecalho.columns[0]: "CHAVE_ACESSO"}, inplace=True)
df_itens.rename(columns={df_itens.columns[0]: "CHAVE_ACESSO"}, inplace=True)

# Merge and write to SQLite
merged_df = pd.merge(df_cabecalho, df_itens, on="CHAVE_ACESSO", how="inner")
conn = sqlite3.connect("notas_fiscais.db")
merged_df.to_sql("notas_fiscais", conn, if_exists="replace", index=False)
conn.commit()

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------- Helper Functions --------------------


def get_schema_description() -> str:
    conn = sqlite3.connect("notas_fiscais.db")
    cursor = conn.execute("PRAGMA table_info(notas_fiscais);")
    return "\n".join([f"{row[1]} ({row[2]})" for row in cursor.fetchall()])


def generate_sql_query(user_question: str, schema: str) -> str:
    prompt = f"""
Você é um assistente de dados. Gere uma consulta SQL válida (SQLite) para responder à pergunta a seguir.

Tabela: notas_fiscais  
Esquema:
{schema}

Pergunta:
{user_question}

Responda somente com a SQL. Não use markdown, não explique.
"""
    response = model.generate_content(prompt)
    sql = response.text.strip()

    # Cleanup possible markdown
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].strip()

    return sql.rstrip(";")


def run_sql_query(sql: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect("notas_fiscais.db")
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        raise RuntimeError(f"Erro ao executar SQL: {e}")


def summarize_results(df: pd.DataFrame, user_question: str) -> str:
    if df.empty:
        return "Nenhum resultado encontrado."

    prompt = f"""
Você é um agente especialista em Notas Fiscais Eletrônicas (NF-e) com amplo conhecimento técnico, fiscal e normativo. Seu objetivo é analisar e validar dados fiscais contidos em dois datasets fornecidos: o dataset "Cabecalhos", que contém informações principais de cada nota fiscal, incluindo a Chave de Acesso (chave primária e identificador único da nota), Número da Nota, Data de Emissão, Valor Total da Nota e demais campos fiscais; e o dataset "Itens", que contém os itens individuais de cada nota fiscal, com informações de Chave de Acesso (chave primária, correspondendo à chave no dataset Cabecalhos), Código do Item, Descrição do Produto, Quantidade, Valor Unitário, Valor Total do Item e demais campos de detalhe.

Sua principal tarefa é realizar a validação de consistência entre os dois datasets: para cada Chave de Acesso, você deve verificar se a soma do campo Valor Total do Item de todos os itens associados corresponde exatamente ao Valor Total da Nota no dataset Cabecalhos. Sempre que encontrar divergências, apresente um relatório informando a Chave de Acesso, o Valor Total informado na Nota, a soma calculada dos itens e a diferença apurada.

Além disso, você deve ser capaz de responder perguntas analíticas e descritivas sobre os dados dos datasets, como: quantidade total de notas fiscais, soma total de valores de notas, número de itens em determinada nota, identificação das maiores notas fiscais emitidas e listagem dos produtos mais comuns. Sempre que possível, apresente as respostas em forma de tabela, lista ou resumo numérico, conforme for mais adequado.

Você também deve identificar anomalias fiscais, como notas fiscais com valor total zerado, notas fiscais sem itens associados, e itens com valores unitários nulos ou negativos. Suas respostas devem ser técnicas, claras, objetivas e fundamentadas. Sempre explique o raciocínio seguido ao apresentar suas conclusões. Utilize a Chave de Acesso como chave primária para todas as operações de cruzamento de dados. Em caso de ausência de dados ou limitações nos arquivos fornecidos, informe a limitação de forma transparente e prossiga com a análise possível.

### Cabeçalhos das Notas Fiscais:
**CHAVE DE ACESSO, MODELO, SÉRIE, NÚMERO, NATUREZA DA OPERAÇÃO, DATA EMISSÃO, EVENTO MAIS RECENTE, DATA/HORA EVENTO MAIS RECENTE, CPF/CNPJ Emitente, RAZÃO SOCIAL EMITENTE, INSCRIÇÃO ESTADUAL EMITENTE, UF EMITENTE, MUNICÍPIO EMITENTE, CNPJ DESTINATÁRIO, NOME DESTINATÁRIO, UF DESTINATÁRIO, INDICADOR IE DESTINATÁRIO, DESTINO DA OPERAÇÃO, CONSUMIDOR FINAL, PRESENÇA DO COMPRADOR, VALOR NOTA FISCAL**

### Itens das Notas Fiscais:
**CHAVE DE ACESSO,MODELO,SÉRIE,NÚMERO,NATUREZA DA OPERAÇÃO,DATA EMISSÃO,CPF/CNPJ Emitente,RAZÃO SOCIAL EMITENTE,INSCRIÇÃO ESTADUAL EMITENTE,UF EMITENTE,MUNICÍPIO EMITENTE,CNPJ DESTINATÁRIO,NOME DESTINATÁRIO,UF DESTINATÁRIO,INDICADOR IE DESTINATÁRIO,DESTINO DA OPERAÇÃO,CONSUMIDOR FINAL,PRESENÇA DO COMPRADOR,NÚMERO PRODUTO,DESCRIÇÃO DO PRODUTO/SERVIÇO,CÓDIGO NCM/SH,NCM/SH (TIPO DE PRODUTO),CFOP,QUANTIDADE,UNIDADE,VALOR UNITÁRIO,VALOR TOTAL**

Ambos os dataframes possuem a coluna **CHAVE DE ACESSO** que é uma chave única de 44 dígitos que identifica cada nota fiscal. Faça o join entre eles usando essa chave para unir as informações de cabeçalho e itens.

Com base nessas informações, responda às perguntas do usuário.

Pergunta:
{user_question}

Resultados:
{df.to_string(index=False)}

Responda em português, com linguagem clara, técnica e objetiva. Não explique o raciocínio, apenas forneça a resposta direta.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------- Chat Loop --------------------


def run_chat_agent():
    schema = get_schema_description()
    print("🤖 Assistente de Notas Fiscais iniciado. Faça sua pergunta ou digite 'sair' para encerrar.")

    while True:
        user_question = input("\n🗨️ Você: ")
        if user_question.lower() in ["sair", "exit", "quit"]:
            print("👋 Encerrando. Até logo!")
            break

        try:
            sql = generate_sql_query(user_question, schema)
            print(f"\n🧠 SQL gerada: {sql}\n")
            df = run_sql_query(sql)
            answer = summarize_results(df, user_question)
            print(f"\n🤖 Resposta: {answer}\n")
        except Exception as e:
            print(f"\n⚠️ Erro: {e}\n")


# 🔹 Entry Point
if __name__ == "__main__":
    run_chat_agent()
