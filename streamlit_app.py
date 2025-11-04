import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import warnings
import re
from scipy.stats import pearsonr, kruskal, mannwhitneyu

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Dashboard Burnout Docente")

# --- INJEÇÃO DE CSS (Fonte 24px) ---
font_size_tabela = "24px"
st.markdown(f"""
<style>
/* CSS para fonte da tabela */
.stDataFrame table th {{
    font-size: {font_size_tabela} !important;
    font-weight: bold !important;
    padding: 10px 5px !important;
}}
.stDataFrame table td {{
    font-size: {font_size_tabela} !important;
    padding: 10px 5px !important;
}}
.stTable table th,
.stTable table td
{{
    font-size: {font_size_tabela} !important;
    padding: 10px 5px !important;
}}
</style>
""", unsafe_allow_html=True)
# --- FIM CSS ---

st.title("A RELAÇÃO ENTRE A CARGA DE TRABALHO DOCENTE E O BURNOUT")
st.markdown("Análise Interativa do Equilíbrio entre Demanda e Bem-estar")
st.markdown("Hiro, Rafaela e Maria")


# --- Função SIMPLES para carregar dados JÁ LIMPOS do disco ---
@st.cache_data
def load_cleaned_data_from_disk(filepath='cleaned_data.csv'):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';')
    except FileNotFoundError:
        st.error(f"Erro Crítico: O arquivo '{filepath}' não foi encontrado.")
        st.info("Execute a Célula 2 (Limpeza) primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return None
    return df

# --- Função para gerar Gráficos de Correlação (Pearson) ---
def plotar_correlacao(df_data, var_x, var_y, titulo, xlabel, ylabel, cor_linha="red"):
    # (O código do Pearson permanece o mesmo)
    try:
        df_plot = df_data[[var_x, var_y]].dropna()
        df_plot[var_x] = pd.to_numeric(df_plot[var_x], errors='coerce')
        df_plot[var_y] = pd.to_numeric(df_plot[var_y], errors='coerce')
        df_plot = df_plot.dropna()
        N = len(df_plot)
        if N < 3:
            st.warning(f"Não há dados suficientes (N < 3) para plotar: {titulo}")
            return
        r, p_value = pearsonr(df_plot[var_x], df_plot[var_y])
        r_abs = abs(r)
        if p_value < 0.001: p_text_display = "ALTAMENTE SIG. (< 0.001)"
        elif p_value < 0.05: p_text_display = f"SIG. ({p_value:.4f})"
        else: p_text_display = f"NÃO SIGNIFICATIVO ({p_value:.4f})"
        if r_abs >= 0.6: forca = "FORTE"
        elif r_abs >= 0.3: forca = "MODERADA"
        elif r_abs >= 0.1: forca = "FRACA"
        else: forca = "NULA"
        direcao = "Positiva" if r > 0 else "Negativa" if r < 0 else "Nula"
        st.markdown("---"); st.markdown(f"### Análise de Correlação: {titulo}")
        col1, col2 = st.columns(2)
        with col1: st.metric(label="Nº Participantes (N)", value=N)
        with col2: st.metric(label="Coeficiente Pearson (r)", value=f"{r:.4f}")
        st.markdown(f"**Significância (P-value):** {p_text_display}")
        st.markdown(f"**Força e Direção:** {forca} ({direcao})")
        st.markdown("#### Visualização da Regressão")
        x_jitter = 0.2 if df_plot[var_x].nunique() < 10 and df_plot[var_x].min() >= 1 and df_plot[var_x].max() <= 5 else 0
        y_jitter = 0.2 if df_plot[var_y].nunique() < 10 and df_plot[var_y].min() >= 1 and df_plot[var_y].max() <= 5 else 0
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df_plot, x=var_x, y=var_y, line_kws={"color": cor_linha, "linewidth": 2}, scatter_kws={"alpha": 0.4}, x_jitter=x_jitter, y_jitter=y_jitter, ax=ax)
        ax.set_title(titulo, fontsize=16); ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6); st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico '{titulo}': {e}")
        st.code(f"Erro detalhado: {e}")

# --- Função para gerar Boxplots (Mann-Whitney U / Kruskal-Wallis) (CORRIGIDA) ---
def plotar_boxplot(df_data, var_x_coluna, titulo, xlabel):
    try:
        # 1. PREPARAÇÃO DOS DADOS
        df_plot = df_data[['ET', var_x_coluna]].copy()
        df_plot['ET'] = pd.to_numeric(df_plot['ET'], errors='coerce')
        if not pd.api.types.is_numeric_dtype(df_plot[var_x_coluna]):
            df_plot[var_x_coluna] = df_plot[var_x_coluna].astype(str)
        df_plot = df_plot.dropna(subset=['ET', var_x_coluna])

        # 2. MAPEAR RÓTULOS (CRIA UMA NOVA COLUNA 'Grupo_Label' PARA O GRÁFICO)
        df_plot_final = df_plot.copy()
        mapa_rotulos = {}
        teste_nome = "Não Aplicável"
        order = None # Define a ordem do gráfico

        if var_x_coluna in ['b4_4_violencia_trabalho', 'b4_3_cultura_feedback', 'b3_3_nivel_ensino_Superior', 'b3_3_nivel_ensino_Anos_Iniciais', 'b2_1_acompanhamento_agrupado']:
            mapa_rotulos = {0.0: 'Não', 1.0: 'Sim'}
            teste_nome = "Mann-Whitney U"
            order = ['Não', 'Sim'] # Ordem Lógica
            df_plot_final['Grupo_Label'] = df_plot_final[var_x_coluna].map(mapa_rotulos).astype(str)

        elif var_x_coluna == 'b1_2_genero':
            df_plot_final = df_plot_final[df_plot_final[var_x_coluna].isin(['Feminino', 'Masculino'])]
            df_plot_final['Grupo_Label'] = df_plot_final[var_x_coluna]
            teste_nome = "Mann-Whitney U"
            order = ['Feminino', 'Masculino'] # Ordem Lógica

        elif var_x_coluna == 'b2_1_acompanhamento_saude': # 4 Grupos
            mapa_rotulos = {0.0: 'Não', 1.0: 'Só Psicológico', 2.0: 'Só Psiquiátrico', 3.0: 'Ambos'}
            df_plot_final['Grupo_Label'] = df_plot_final[var_x_coluna].map(mapa_rotulos).astype(str)
            teste_nome = "Kruskal-Wallis H"
            order = ['Não', 'Só Psicológico', 'Só Psiquiátrico', 'Ambos']

        # <<< CORREÇÃO AQUI: Modificado o teste de Tipo de Instituição >>>
        elif var_x_coluna == 'b3_7_grupo_instituicao':
            # Filtra ANTES de mapear, mantendo apenas 0.0 (Pública) e 1.0 (Privada)
            grupos_validos_num = [0.0, 1.0]
            df_plot_final = df_plot_final[df_plot_final[var_x_coluna].isin(grupos_validos_num)]

            mapa_rotulos = {0.0: 'Somente Pública', 1.0: 'Somente Privada'} # Mapa reduzido
            df_plot_final['Grupo_Label'] = df_plot_final[var_x_coluna].map(mapa_rotulos).astype(str)
            teste_nome = "Mann-Whitney U" # Agora é Mann-Whitney U
            order = ['Somente Pública', 'Somente Privada'] # Ordem de 2 grupos
        # <<< FIM DA CORREÇÃO >>>

        else: # Fallback
            df_plot_final['Grupo_Label'] = df_plot_final[var_x_coluna].astype(str)
            order = sorted(df_plot_final['Grupo_Label'].unique())

        df_plot_final = df_plot_final.dropna(subset=['Grupo_Label'])

        # 3. EXTRAÇÃO DE GRUPOS E TESTE ESTATÍSTICO
        grupos_dados_et = [df_plot_final[df_plot_final['Grupo_Label'] == nome]['ET'].values for nome in order if not df_plot_final[df_plot_final['Grupo_Label'] == nome].empty]
        N = len(df_plot_final)
        K = len(grupos_dados_et) # Usa o número de grupos que realmente têm dados

        if K < 2 or N < 3:
            st.warning(f"Não há dados suficientes (N < 3 ou K < 2) para plotar o Boxplot para: {titulo}. N={N}, K={K}")
            return

        # 4. TESTE ESTATÍSTICO
        if K == 2:
            statistic, p_value = mannwhitneyu(grupos_dados_et[0], grupos_dados_et[1], alternative='two-sided')
        else:
            statistic, p_value = kruskal(*grupos_dados_et)

        # 5. DEFINIR SIGNIFICÂNCIA E EXIBIR MÉTRICAS
        if p_value < 0.001: p_text = "ALTAMENTE SIG. (< 0.001)"
        elif p_value < 0.05: p_text = f"SIG. ({p_value:.4f})"
        else: p_text = f"NÃO SIGNIFICATIVO ({p_value:.4f})"

        st.markdown("---"); st.markdown(f"### Comparação Categórica: {titulo}")
        col1, col2 = st.columns(2)
        with col1: st.metric(label="Nº Participantes (N)", value=N)
        with col2: st.metric(label=f"Estatística ({teste_nome})", value=f"{statistic:.4f}")
        st.markdown(f"**Significância (P-value):** {p_text}")

        # 6. PLOTAR O BOXPLOT
        st.markdown("#### Distribuição por Grupos")
        plt.figure(figsize=(K * 3.5, 7))
        sns.boxplot(data=df_plot_final, x='Grupo_Label', y='ET', palette="Pastel1", order=order)
        sns.stripplot(data=df_plot_final, x='Grupo_Label', y='ET', color=".3", alpha=0.5, jitter=0.1, order=order)
        plt.title(f"{titulo}", fontsize=16); plt.xlabel(xlabel, fontsize=12); plt.ylabel('Escore Total de Indício de Burnout (20-100)', fontsize=12)
        plt.xticks(rotation=15, ha='right'); plt.grid(True, linestyle='--', alpha=0.6, axis='y'); plt.tight_layout(); st.pyplot(plt)

    except Exception as e:
        st.error(f"Erro ao gerar Boxplot '{titulo}': {e}")
        st.code(f"Erro detalhado: {e}")

# --- Código Principal da Dashboard ---
df = load_cleaned_data_from_disk()

if df is not None and not df.empty:
    st.success("Arquivo 'cleaned_data.csv' carregado com sucesso!")

    colunas_necessarias = [
        'b1_2_genero', 'b3_3_nivel_ensino', 'Nivel_Burnout', 'ET',
        'b3_5_carga_horaria', 'b2_3_tempo_energia_lazer', 'b3_9_carga_administrativa',
        'b4_7_apoio_gestao_escolar', 'b4_5_intencao_abandonar_profissao', 'b1_1_idade',
        'b2_2_frequencia_autocuidado',
        'b3_7_grupo_instituicao', # Coluna numérica para instituição
        'b2_1_acompanhamento_saude', 'b2_1_acompanhamento_agrupado', # Ambas colunas de saúde
        'b4_4_violencia_trabalho', 'b4_3_cultura_feedback', 'b3_2_tempo_profissao'
    ]
    colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]

    if colunas_faltando:
        st.error(f"Erro CSV: Colunas essenciais ausentes: {', '.join(colunas_faltando)}")
    else:
        df['Nivel_Burnout'] = df['Nivel_Burnout'].astype(str)
        df['b3_3_nivel_ensino'] = df['b3_3_nivel_ensino'].astype(str)

        # --- BARRA LATERAL: Filtros e Seletores ---
        st.sidebar.header("Filtros de Segmentação")
        try:
            generos = ['Todos'] + sorted(list(df['b1_2_genero'].dropna().unique()))
            genero_selecionado = st.sidebar.selectbox("Gênero:", generos)

            all_options = set()
            for item in df['b3_3_nivel_ensino'].dropna().unique():
                parts = [part.strip() for part in re.split(r'\s*;\s*', str(item))]
                all_options.update(parts)
            unique_niveis = sorted([opt for opt in all_options if opt and opt.lower() != 'nan'])

            niveis_selecionados = st.sidebar.multiselect(
                "Nível(is) de Ensino:", options=unique_niveis, default=[]
            )
        except Exception as e_sidebar:
            st.sidebar.error(f"Erro filtros: {e_sidebar}")
            genero_selecionado = 'Todos'; niveis_selecionados = []

        # --- SELETORES DE GRÁFICOS ---
        st.sidebar.header("Análises Estatísticas")

        # Opções de Correlação (Pearson)
        opcoes_correlacao = {
            'Nenhum': None,
            'Idade vs. Burnout': ('b1_1_idade', 'ET', 'Idade e Indício de Burnout', 'Idade (anos)', 'Escore Total de Indício de Burnout (20-100)', 'gray'),
            'Tempo de Profissão vs. Burnout': ('b3_2_tempo_profissao', 'ET', 'Tempo de Profissão e Indício de Burnout', 'Tempo de Profissão (anos)', 'Escore Total de Indício de Burnout (20-100)', 'cyan'),
            'Carga Horária vs. Burnout': ('b3_5_carga_horaria', 'ET', 'Carga Horária e Indício de Burnout', 'Carga Horária Semanal (horas)', 'Escore Total de Indício de Burnout (20-100)', 'red'),
            'Carga Administrativa vs. Burnout': ('b3_9_carga_administrativa', 'ET', 'Carga Administrativa e Indício de Burnout', 'Percepção Carga Administrativa (1-5)', 'Escore Total de Indício de Burnout (20-100)', 'purple'),
            'Apoio da Gestão vs. Burnout': ('b4_7_apoio_gestao_escolar', 'ET', 'Apoio da Gestão e Indício de Burnout', 'Percepção Apoio da Gestão (1-5)', 'Escore Total de Indício de Burnout (20-100)', 'green'),
            'Lazer vs. Burnout': ('b2_3_tempo_energia_lazer', 'ET', 'Lazer e Indício de Burnout', 'Percepção Tempo/Energia p/ Lazer (1-5)', 'Escore Total de Indício de Burnout (20-100)', 'blue'),
            'Autocuidado vs. Burnout': ('b2_2_frequencia_autocuidado', 'ET', 'Autocuidado e Indício de Burnout', 'Frequência de Autocuidado (1-5)', 'Escore Total de Indício de Burnout (20-100)', 'darkgreen'),
            'Burnout vs. Intenção de Abandonar': ('ET', 'b4_5_intencao_abandonar_profissao', 'Burnout e Intenção de Abandonar', 'Escore Total de Indício de Burnout (20-100)', 'Intenção de Abandonar (1-5)', 'orange'),
            'Autocuidado vs. Lazer': ('b2_2_frequencia_autocuidado', 'b2_3_tempo_energia_lazer', 'Correlação entre Autocuidado e Lazer', 'Frequência de Autocuidado (1-5)', 'Percepção Tempo/EnergIA p/ Lazer (1-5)', 'blueviolet')
        }

        # <<< CORREÇÃO AQUI: Ajustado o nome da opção >>>
        opcoes_boxplot = {
            'Nenhum': None,
            'Gênero': ('b1_2_genero', 'Distribuição por Gênero', 'Gênero'),
            'Violência no Trabalho (Sim/Não)': ('b4_4_violencia_trabalho', 'Exposição à Violência', 'Sofreu Violência'),
            'Cultura de Feedback (Sim/Não)': ('b4_3_cultura_feedback', 'Cultura de Feedback', 'Possui Feedback'),
            'Acompanhamento Saúde Mental (Sim/Não)': ('b2_1_acompanhamento_agrupado', 'Acompanhamento Saúde Mental (Agrupado Sim/Não)', 'Faz Acompanhamento'),
            'Acompanhamento Saúde Mental (Psicológico e Psiquiátrico)': ('b2_1_acompanhamento_saude', 'Acompanhamento Saúde Mental (Psicológico e Psiquiátrico)', 'Tipo de Acompanhamento'),
            'Tipo de Instituição (Pública vs. Privada)': ('b3_7_grupo_instituicao', 'Instituição (Pública vs. Privada)', 'Tipo de Instituição'),
            'Atuação no Ensino Superior (Sim/Não)': ('b3_3_nivel_ensino_Superior', 'Atuação no Ensino Superior', 'Leciona no Ens. Superior'),
            'Atuação nos Anos Iniciais (Sim/Não)': ('b3_3_nivel_ensino_Anos_Iniciais', 'Atuação nos Anos Iniciais', 'Leciona nos Anos Iniciais')
        }

        # SELETORES
        grafico_correlacao_selecionado = st.sidebar.selectbox("Correlação (Pearson):", options=opcoes_correlacao.keys())
        st.sidebar.markdown("---")
        grafico_boxplot_selecionado = st.sidebar.selectbox("Comparação de Grupos (Boxplot):", options=opcoes_boxplot.keys())


        # Aplicação dos Filtros
        df_filtrado = df.copy()
        try:
            if genero_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['b1_2_genero'] == genero_selecionado]
            if niveis_selecionados:
                for nivel in niveis_selecionados:
                    df_filtrado = df_filtrado[df_filtrado['b3_3_nivel_ensino'].str.contains(re.escape(nivel), na=False, case=False, regex=True)]
        except Exception as e_filter:
            st.error(f"Erro ao filtrar: {e_filter}")
            df_filtrado = pd.DataFrame()

        # --- Criação de colunas auxiliares para Boxplots (Sim/Não) ---
        if 'b3_3_nivel_ensino' in df_filtrado.columns:
            df_filtrado['b3_3_nivel_ensino_Superior'] = np.where(
                df_filtrado['b3_3_nivel_ensino'].str.contains('Ensino Superior', na=False, case=False, regex=False), 1, 0)
            df_filtrado['b3_3_nivel_ensino_Anos_Iniciais'] = np.where(
                df_filtrado['b3_3_nivel_ensino'].str.contains('Ensino Fundamental  - Anos Iniciais', na=False, case=False, regex=False), 1, 0)

        # Exibir Resultados
        if not df_filtrado.empty:
            st.subheader(f"Resultados (N = {len(df_filtrado)})")

            # ... (Exibição dos Filtros Ativos) ...
            filtros_aplicados = []
            if genero_selecionado != 'Todos': filtros_aplicados.append(f"**Gênero:** {genero_selecionado}")
            if niveis_selecionados:
                niveis_texto = " E ".join(niveis_selecionados)
                filtros_aplicados.append(f"**Nível(is) de Ensino:** {niveis_texto}")
            if filtros_aplicados: st.info(f"Filtros aplicados: {'; '.join(filtros_aplicados)}")
            else: st.info("Mostrando resultados para todos os participantes (nenhum filtro aplicado).")
            st.markdown("---")

            # --- VISUALIZAÇÃO 1: DISTRIBUIÇÃO BURNOUT (Sempre visível) ---
            if 'Nivel_Burnout' in df_filtrado.columns:
                df_validos = df_filtrado[~df_filtrado['Nivel_Burnout'].isin(['Erro', 'Inválido', 'nan'])]
                if not df_validos.empty:
                    st.markdown("### Distribuição Burnout")
                    contagem_niveis = df_validos['Nivel_Burnout'].value_counts()
                    count_n1 = contagem_niveis.get('Nível 1', 0)
                    count_n2 = contagem_niveis.get('Nível 2', 0)
                    count_n3 = contagem_niveis.get('Nível 3', 0)
                    count_n4 = contagem_niveis.get('Nível 4', 0)
                    count_n5 = contagem_niveis.get('Nível 5', 0)

                    st.markdown("##### Frequência (Nº de Professores) por Nível:")
                    st.markdown(f"""
                    <div class="metric-row"><span class="metric-label">Nível 1 (Nenhum Indício)</span><span class="metric-value">{count_n1}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 2 (Possibilidade)</span><span class="metric-value">{count_n2}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 3 (Fase Inicial)</span><span class="metric-value">{count_n3}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 4 (Instalação)</span><span class="metric-value metric-value-n4">{count_n4}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 5 (Fase Considerável)</span><span class="metric-value metric-value-n5">{count_n5}</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <style>
                    /* ... (Seu código CSS de fonte 24px) ... */
                    /* AQUI ESTÁ O CÓDIGO QUE FALTA */
                    /* CSS para Tabela Vertical Customizada */
                    .metric-row {{
                        display: flex; flex-direction: row; justify-content: space-between; 
                        align-items: center; border-bottom: 1px solid #DDDDDD; 
                        padding: 12px 5px; 
                    }}
                    .metric-label {{
                        font-size: 20px; color: #333333; /* Cor que faltava */
                    }}
                    .metric-value {{
                        font-size: 26px; font-weight: 600; color: #333333; 
                    }}
                    .metric-value-n4 {{ color: #FF9800 !important; }}
                    .metric-value-n5 {{ color: #F44336 !important; }}
                    </style>
                    """, unsafe_allow_html=True)


                    # Gráfico de Barras de Distribuição
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5)); cores = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336', '#B71C1C']
                        ordem_niveis = ['Nível 1', 'Nível 2', 'Nível 3', 'Nível 4', 'Nível 5']
                        contagem_ordenada = contagem_niveis.reindex(ordem_niveis, fill_value=0)
                        bars = ax.bar(contagem_ordenada.index, contagem_ordenada.values, color=cores)
                        ax.set_title(f"Distribuição (N={len(df_validos)})"); ax.set_xlabel("Nível Risco"); ax.set_ylabel("Contagem")
                        total_validos = len(df_validos)
                        custom_labels = [f"{count} ({((count/total_validos)*100):.1f}%)" if total_validos > 0 else f"{count} (0.0%)" for count in contagem_ordenada]
                        ax.bar_label(bars, labels=custom_labels, label_type='edge', padding=3, fontsize=9); st.pyplot(fig)
                    except Exception as e_plot: st.error(f"Gráfico: {e_plot}")

                    # Métrica
                    try:
                        risco45 = contagem_ordenada.get('Nível 4', 0) + contagem_ordenada.get('Nível 5', 0)
                        perc45 = (risco45 / len(df_validos)) * 100 if len(df_validos) > 0 else 0
                        st.metric("Risco Alto/Crítico (Níveis 4/5)", f"{perc45:.1f}%", f"Total: {risco45}", delta_color="inverse")
                    except Exception as e_metric: st.error(f"Métrica: {e_metric}")
                else: st.warning("Sem dados válidos de Nível Burnout.")
            else: st.error("Coluna 'Nivel_Burnout' não encontrada.")

            st.markdown("---")

            # --- VISUALIZAÇÃO 2: GRÁFICOS DINÂMICOS ---
            if grafico_correlacao_selecionado != 'Nenhum':
                params = opcoes_correlacao[grafico_correlacao_selecionado]
                var_x, var_y, titulo, xlabel, ylabel, cor = params
                plotar_correlacao(df_filtrado, var_x, var_y, titulo, xlabel, ylabel, cor)

            if grafico_boxplot_selecionado != 'Nenhum':
                 params = opcoes_boxplot[grafico_boxplot_selecionado]
                 var_x_coluna, titulo, xlabel = params
                 plotar_boxplot(df_filtrado, var_x_coluna, titulo, xlabel)

        else:
            st.info("Nenhum professor corresponde aos filtros selecionados.")
else:
    st.info("👈 Carregue o arquivo CSV LIMPO ('cleaned_data.csv') na barra lateral para iniciar.")
