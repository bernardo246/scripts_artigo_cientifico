# scripts_artigo_cientifico

Script para comparar desempenho de processamento analítico em três formatos:
- CSV (texto)
- JSONL (texto)
- Parquet (colunar)

## Por que usar
- Mostrar, com a mesma carga lógica, a diferença de tempo, CPU, memória e throughput entre formatos.
- Gerar métricas objetivas para embasar discussão em artigo científico.
- Reproduzir experimentos com volume e número de arquivos configuráveis.

## Como usar
1. Instale as dependências:
   - `numpy`
   - `pandas`
   - `psutil`
   - `pyarrow`
2. Execute:
   - `python3 workload_datalake_comparacao_v2.py`
3. Opcionalmente ajuste volume:
   - `python3 workload_datalake_comparacao_v2.py --rows 2000000 --files 16`

O script gera dados sintéticos, converte para Parquet e imprime os resultados comparativos no terminal.
