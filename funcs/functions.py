import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def leverage(x):
    if np.can_cast(type(x), np.ndarray):
        x = np.array(x)
    else:
        raise TypeError
    return 1/x.size + np.square(x - x.mean())/np.sum(np.square(x - x.mean()))



def check_leverage(leverage, essay, critical=3, n_param=2):
    """Esta função verifica se o leverage é maior do que o valor estabelecido
    """
    if np.can_cast(type(leverage), np.ndarray):
        leverage = np.array(leverage)
    else:
        raise TypeError
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError        
        
    critical = critical*n_param/leverage.size
    aux = True
    for x, ens in zip(leverage, essay):
        if x > critical:
            aux = False
            print(f'A alavancagem ({round(x, 3)}) do ensaio "{ens}" esta acima do limite esperado ({critical})')
    if aux:
        print(f"Nenhum ponto com leverage acima de {critical} foi encontrado")
        
        
def make_leverage_plot(concentration, leverage, n_param=2, mild=2, extreme=3):
    if np.can_cast(type(leverage), np.ndarray):
        leverage = np.array(leverage)
    else:
        raise TypeError
        
    if np.can_cast(type(concentration), np.ndarray):
        concentration = np.array(concentration)
    else:
        raise TypeError     
        
    mild = mild*n_param/leverage.size
    extreme = extreme*n_param/leverage.size
    
    plt.figure(figsize=(6,3))

    plt.scatter(concentration, leverage, color='k', facecolor='none', label="Alavancagem")
    plt.axhline(y=extreme, color="r", linestyle="--", label="Extremo")
    plt.axhline(y=mild, color="y", linestyle="--", label="Mild")
    plt.ylim(bottom=0)

    plt.ylabel("Alavancagem")
    plt.xlabel("Concentração")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    
def make_f_distribution_plot(Fcalc, p_valor, gl_numerator, gl_denominator, alpha=0.05):
    
    Ftab = stats.f.ppf(1-alpha, gl_numerator, gl_denominator)
    
    # Verificando qual é o maior valor para adaptar o intervalo do eixo x
    if Fcalc > Ftab:
        x_max = Fcalc*1.05 + 1
    else:
        x_max = Ftab*1.05 + 5

    # obtendo valores de x para o gráfico
    x = np.linspace(0, x_max, 1000)

    # criando uma instância para a distribuição F com grau de liberdade do teste
    f1 = stats.f(gl_numerator, gl_denominator, 0)

    # plotando o gráfico
    # criando o canvas
    plt.figure(figsize=(6, 3))
    # adicionando a linha com valores teórios
    legend = '$gl_{1}$ = ' + str(gl_numerator) + '; $gl_{2}$ = ' + str(gl_denominator)
    plt.plot(x, f1.pdf(x), label = legend, color = 'black')

    # adicionando a linha com o valor crítico
    legend = '$F_{crítico} = $' + str(round(Ftab, 2))
    plt.axvline(Ftab, 0, 1, label=legend, color='blue', ls="--")

    # adicionando o ponto com o valor da estatística do teste
    legend = '$F_{calc} = $' + str(round(Fcalc, 2))
    plt.scatter(Fcalc, 0, color='red', label=legend)

    # preenchendo a área do p-valor
    x = np.linspace(Fcalc, int(x_max), 1000)
    if p_valor < 0.0001:
        label = "$p-valor<0.001$"
    else:
        label = "$p-valor=" + f"{round(p_valor, 3)}" + "$"
        
    plt.fill_between(x, f1.pdf(x), label = label, color = 'salmon', alpha=.5)

    # ajustando o gráfico
    plt.xlim(-.1, x_max + 0.1)
    plt.ylim(bottom=0.0)
    plt.xlabel('$F(x)$')
    plt.ylabel('Densidade')
    plt.legend(fontsize=8)
    plt.show() 
    
    
def make_t_distribution_plot(t_calc, gl, alpha=0.05):
    
    t_critico = stats.t.ppf(1-alpha/2, gl)
    
    if np.abs(t_calc) < t_critico:
        x_lim = 5
    else:
        x_lim = np.abs(t_calc)*1.2

    # criando o canvas
    fig, ax1 = plt.subplots(figsize=(8,4))

    # criando valores para a distribuição t de Student com gl
    x = np.linspace(-1*x_lim, x_lim, 1000)
    y = stats.t.pdf(x, gl)
    label = "Dist. t-Student ($gl=$" + str(gl) + ")"
    ax1.plot(x, y, c="k", label=label)

    # adicionando linhas com o valor crítico do teste
    label= '$t_{crítico} = |$' + str(round(t_critico, 3)) + '$|$'
    ax1.axvline(stats.t.ppf(1-alpha/2, gl), 0, 1, label=label, color='blue', ls='--')
    ax1.axvline(stats.t.ppf(alpha/2, gl), 0, 1, color='blue', ls='--')

    # adicionando o valor da estatística do teste
    label = "$t_{calc} = $" + str(round(t_calc, 3))
    ax1.vlines(t_calc, 0, stats.t.pdf(np.abs(t_calc), gl, loc=0, scale=1), label=label, color='red')

    # preenchendo o lado esquero
    x = np.linspace(-1*x_lim, -1*abs(t_calc), 1000)
    ax1.fill_between(x, stats.t.pdf(x, gl), label = '$probabilidade$', color = 'salmon')

    # preenchendo o lado direito
    x = np.linspace(abs(t_calc), x_lim, 1000)
    ax1.fill_between(x,stats.t.pdf(x, gl, loc=0, scale=1), color = 'salmon')

    plt.xlim(-1*x_lim - .1, x_lim + .1)
    plt.ylim(bottom=0.0)
    plt.legend(loc=1)
    plt.show()
    
    
def make_residuals_plot_acquisition_order(residuals, aquisition_order):

    if np.can_cast(type(residuals), np.ndarray):
        residuals = np.array(residuals)
    else:
        raise TypeError    

    if np.can_cast(type(aquisition_order), np.ndarray):
        aquisition_order = np.array(aquisition_order)
    else:
        raise TypeError            
    
    idx   = np.argsort(aquisition_order)

    aquisition_order = np.array(aquisition_order)[idx]
    residuals = np.array(residuals)[idx]


    # obtendo maior resíduo (absoluto) para utilizar como y_min
    y_min = np.max([np.abs(residuals.min()), np.abs(residuals.max())])

    # criando o canvas
    plt.figure(figsize=(6,4))
    # adicionando os resíduos
    plt.scatter(aquisition_order, residuals, edgecolors='k', facecolor='None', label="Resíduos")
    plt.plot(aquisition_order, residuals, c='k')
    plt.xlabel("Ordem de coleta")
    plt.ylabel("Resíduos")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.ylim(bottom= -1.1*y_min, top = 1.1*y_min )    
    plt.xticks(aquisition_order)
    plt.show()
    
    
def make_independent_residuals_plot(residuals, predicted):

    if np.can_cast(type(residuals), np.ndarray):
        residuals = np.array(residuals)
    else:
        raise TypeError    

    if np.can_cast(type(predicted), np.ndarray):
        predicted = np.array(predicted)
    else:
        raise TypeError            
        
    ymin = y_min = np.max([np.abs(residuals.min()), np.abs(residuals.max())])
    
    plt.figure(figsize=(6,4))
    plt.scatter(predicted, residuals, edgecolors='k', facecolor='None')
    plt.xlabel("Sinal predito")
    plt.ylabel("Resíduos")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.ylim(bottom= -1.1*y_min, top = 1.1*y_min )    

    plt.show()
    
def standardized_residuals(residual, MSQE):
    if np.can_cast(type(residual), np.ndarray):
        residual = np.array(residual)
    else:
        raise TypeError      
            
    return residual/np.sqrt(MSQE)

def check_standardized_residuals(residual_standardized, essay, limite=3):
    
    if np.can_cast(type(residual_standardized), np.ndarray):
        residual_standardized = np.array(residual_standardized)
    else:
        raise TypeError      
        
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError              
        
        
    aux = True
    for res_stand, ens in zip(residual_standardized, essay):
        if np.abs(res_stand) > limite:
            aux = False
            print(f'O resíduo padronizado ({round(res_stand, 3)}) do ensaio "{ens}" esta acima do limite esperado ({limite})')
    if aux:
        print(f"Nenhum resíduo padronizado é maior do que {limite}.")
        
        
        
def make_standardized_residuals(standardized_residuals, y_pred, mild=2, extreme=3):

    if np.can_cast(type(standardized_residuals), np.ndarray):
        standardized_residuals = np.array(standardized_residuals)
    else:
        raise TypeError    

    if np.can_cast(type(y_pred), np.ndarray):
        y_pred = np.array(y_pred)
    else:
        raise TypeError    

    
    # determinando o limite do eixo y
    y_min = np.abs(np.max([np.abs(standardized_residuals)]))
    if y_min < extreme:
        y_min = extreme*1.2
    else:
        y_min = y_min*1.2

    # criando o canvas
    plt.figure(figsize=(6,3))
    # adicionando os dados studentizados
    plt.scatter(y_pred, standardized_residuals, label="Resíduos Padronizados",  color='k', facecolor='none')

    # linhas em y = 0
    plt.axhline(y=0, color="gray", linestyle=(0, (2, 3)))

    # limite moderado
    plt.axhline(y=mild, color="orange", label="Limite moderado", linestyle="dotted")
    plt.axhline(y=-1*mild, color="orange", linestyle="dotted")
    
    # limite extremo
    plt.axhline(y=extreme, color="red", label="Limite extremo", linestyle="dashed")
    plt.axhline(y=-1*extreme, color="red", linestyle="dashed")

    # minor
    plt.ylabel("Resíduos Padronizados")
    plt.xlabel("Sinal predito")
    plt.ylim(-1*y_min, y_min)
    plt.legend(bbox_to_anchor=(1.01,1))
    plt.show()
    
    
def studentized_residuals(residuals, MSQE, leverage):
    if np.can_cast(type(residuals), np.ndarray):
        residuals = np.array(residuals)
    else:
        raise TypeError       
        
    if np.can_cast(type(leverage), np.ndarray):
        leverage = np.array(leverage)
    else:
        raise TypeError       
        
        
    return residuals/np.sqrt(MSQE*(1 - leverage))


def check_studentized_residuals(residuos_student, essay, limite=2):
    
    if np.can_cast(type(residuos_student), np.ndarray):
        residuos_student = np.array(residuos_student)
    else:
        raise TypeError       
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError   
        
        
    aux = True
    for res_stu, ens in zip(residuos_student, essay):
        if np.abs(res_stu) > limite:
            aux = False
            print(f'O resíduo studentizado ({round(res_stu, 3)}) do ensaio "{ens}" esta acima do limite esperado ({limite})')
    if aux:
        print(f"Nenhum resíduo studentizado é maior do que {limite}.")
        
def make_studentized_residuals_plot(studentized_residuals, y_pred, mild=2, extreme=3):
    
    # determinando o limite do eixo y
    y_min = np.abs(np.max([np.abs(studentized_residuals)]))
    if y_min < extreme:
        y_min = extreme*1.2
    else:
        y_min = y_min*1.2

    # criando o canvas
    plt.figure(figsize=(6,3))
    # adicionando os dados studentizados
    plt.scatter(y_pred, studentized_residuals, label="Resíduos Studentizados",  color='k', facecolor='none')

    # linhas em y = 0
    plt.axhline(y=0, color="gray", linestyle=(0, (2, 3)))

    # limite moderado
    plt.axhline(y=mild, color="orange", label="Limite moderado", linestyle="dotted")
    plt.axhline(y=-1*mild, color="orange", linestyle="dotted")
    
    # limite extremo
    plt.axhline(y=extreme, color="red", label="Limite extremo", linestyle="dashed")
    plt.axhline(y=-1*extreme, color="red", linestyle="dashed")

    # minor
    plt.ylabel("Resíduos Studentizados")
    plt.xlabel("Sinal predito")
    plt.ylim(-1*y_min, y_min)
    plt.legend(bbox_to_anchor=(1.01,1))
    plt.show()
    
    
def deleted_t_residuals(residuals, leverage, MSQE, n_param=2):
        
    if np.can_cast(type(residuals), np.ndarray):
        residuals = np.array(residuals)
    else:
        raise TypeError         
        
    if np.can_cast(type(leverage), np.ndarray):
        leverage = np.array(leverage)
    else:
        raise TypeError                 
        
    ri = studentized_residuals(residuals=residuals, leverage=leverage, MSQE=MSQE)
    ti = ri*np.sqrt((ri.size - n_param - 1)/(ri.size - n_param - np.square(ri)))
    return ti


def check_deleted_t_residuals(ti, essay, n_param=2, alpha=0.05):
    
    if np.can_cast(type(ti), np.ndarray):
        ti = np.array(ti)
    else:
        raise TypeError      
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError              
    
    gl = ti.size - 1 - n_param
    t_critico = stats.t.ppf(1-alpha/2, gl)
    
    p_value = []
    aux = True
    for t, ens in zip(ti, essay):
        p_value.append((1 - stats.t.cdf(np.abs(t), gl))*2)
        
        if np.abs(t) > t_critico:
            aux = False
            print(f'Rejeita H0, e o ponto obtido no ensaio "{ens}" é considerado um possível outlier (com {100*(1-alpha)}% de confiança)')
        # else:
        #     print(f'Falha em rejeitar H0, e o ponto obtido no ensaio "{ens}" não é considerado um outlier (com {100*(1-alpha)}% de confiança)')
    
    if aux:
        print(f"Nenhum resíduo studentizado é maior do que {t_critico}.")    
    
    return t_critico, p_value


def make_deleted_t_residuals(ti, y_pred, n_param=2, alpha_mild=0.05, alpha_extreme=None):
    
    gl = ti.size - 1 - n_param
    
    if np.can_cast(type(ti), np.ndarray):
        ti = np.array(ti)
    else:
        raise TypeError    
        
    if np.can_cast(type(y_pred), np.ndarray):
        y_pred = np.array(y_pred)
    else:
        raise TypeError            
    
    # obtendo o valor crítico
    t_critico_mild = stats.t.ppf(1-alpha_mild/2, gl)
    
    if alpha_extreme is None:
        t_critico_extreme = t_critico_mild    
    else:
        t_critico_extreme = stats.t.ppf(1-alpha_extreme/2, gl)
    
    # determinando o limite do eixo y
    y_min = np.abs(np.max([np.abs(ti)]))

    if y_min < t_critico_extreme:
        y_min = t_critico_extreme*1.2
    else:
        y_min = y_min*1.2

    # criando o canvas
    plt.figure(figsize=(6,3))
    # adicionando os dados studentizados
    plt.scatter(y_pred, ti, label="Resíduos t deletados",  color='k', facecolor='none')

    # linhas em y = 0
    plt.axhline(y=0, color="gray", linestyle="-")

    # limite moderado
    plt.axhline(y=t_critico_mild, color="orange", label="Limite moderado", linestyle="dotted")
    plt.axhline(y=-1*t_critico_mild, color="orange", linestyle="dotted")
    
    # limite extremo
    if alpha_extreme is not None:
        plt.axhline(y=t_critico_extreme, color="red", label="Limite extremo", linestyle="dashed")
        plt.axhline(y=-1*t_critico_extreme, color="red", linestyle="dashed")

    # minor
    plt.ylabel("Resíduos Studentizados")
    plt.xlabel("Sinal predito")
    plt.ylim(-1*y_min, y_min)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    
def cook_distance(studentized_residuals, leverage, n_param=2):

    if np.can_cast(type(studentized_residuals), np.ndarray):
        studentized_residuals = np.array(studentized_residuals)
    else:
        raise TypeError  
        
    if np.can_cast(type(leverage), np.ndarray):
        leverage = np.array(leverage)
    else:
        raise TypeError          
    
    
    cook = (np.square(studentized_residuals)/(n_param))*(leverage/(1-leverage))
    
    return cook


def check_cooks_distance(cook, essay, influential=1):
    
    if np.can_cast(type(cook), np.ndarray):
        cook = np.array(cook)
    else:
        raise TypeError      
        
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError              
        
        
    aux = True
    for ck, ens in zip(cook, essay):
        if np.abs(ck) > influential:
            aux = False
            print(f'A distância de Cook ({round(ck, 3)}) do ensaio "{ens}" é influente (c > {influential})')
    if aux:
        print(f"Nenhuma amostra foi considerada influente ({influential}).")
        
        
def make_cook_distance_plot(cook, essay, influential_mild=0.5, influential_extreme=1.0):
    
    if np.can_cast(type(cook), np.ndarray):
        cook = np.array(cook)
    else:
        raise TypeError    
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError            

    plt.figure(figsize=(8,4))
    plt.bar(essay, cook, color="lightgray", edgecolor="k")
    plt.axhline(y=influential_mild, color="orange", label="Limite moderado", linestyle="dotted")
    plt.axhline(y=influential_extreme, color="red", label="Limite extremo", linestyle="dotted")

    plt.xticks(rotation=90)
    plt.ylabel("Cook's distance")
    plt.show()
    
    
def make_slopes_versus_intercepts_plot(x, y, essay):

    # cheking
    if np.can_cast(type(x), np.ndarray):
        x = np.array(x)
    else:
        raise TypeError      
        
    if np.can_cast(type(y), np.ndarray):
        y = np.array(y)
    else:
        raise TypeError    
        
    if np.can_cast(type(essay), np.ndarray):
        essay = np.array(essay)
    else:
        raise TypeError            
    
    # linear regression with full dataset
    reg = stats.linregress(x, y)
    b1_full = reg[0]
    b0_full = reg[1]

    # linear regression with reduced dataset
    size = x.size - 1
    b1_reduced = []
    b0_reduced = []

    for i in range(x.size):
        new_data_x = np.delete(x, i)
        new_data_y = np.delete(y, i)

        reg = stats.linregress(new_data_x, new_data_y)
        b1_reduced.append(reg[0])
        b0_reduced.append(reg[1])

    # plottin
    plt.figure(figsize=(8,6))
    plt.scatter(b0_reduced, b1_reduced, label="Dataset reduzido", color="k", facecolor="none")
    plt.scatter(b0_full, b1_full, label="Dataset completo", c="r")
    
    for i in range(len(b0_reduced)):
        plt.annotate(essay[i], (b0_reduced[i], b1_reduced[i]))

    b0_reduced.append(b0_full)
    x_min = min(b0_reduced) - 1.1*min(b0_reduced)
    x_max = max(b0_reduced)*1.1

    left, right = plt.xlim()
    x_lim = np.max(np.abs([left, right]))
    plt.xlim(left-.1*x_lim, right+.1*x_lim)

    bottom, top = plt.ylim()
    y_lim = np.max(np.abs([bottom, top]))
    plt.ylim(bottom-.1*y_lim, top+.1*y_lim)

    plt.legend()
    plt.xlabel("Intercepto")
    plt.ylabel("Coeficiente angular")
    plt.show()
    
    
def fit(df_data, alpha=0.05):
    columns = df_data.columns
    x_name, y_name = columns[0], columns[1]
    
    # realizando o ajuste linear
    regression = stats.linregress(df_data[x_name], df_data[y_name])
    b1 = regression[0]
    b0 = regression[1]
    
    # calculando valores preditos
    df_data["y_pred"] = b0 + b1*df_data[x_name]
    
    # calculando resíduos
    df_data["Resíduos"] = df_data[y_name] - df_data["y_pred"]
    
    # calculando SQE
    SQE = np.square(df_data["Resíduos"]).sum()
    gl_erro = df_data.shape[0] - 2
    MSQE = SQE/gl_erro
    
    # t critico
    t_critico = stats.t.ppf(1-alpha/2, gl_erro)
    
    # calculando b1 interval
    Sxx = np.square(df_data[x_name] - df_data[x_name].mean()).sum()
    b1_std = np.sqrt(MSQE/Sxx)
    b1_ic = t_critico*b1_std
    print(MSQE)
    print(Sxx)
    
    # b1 significance test
    t_b1 = b1/b1_std
    p_valor_b1 = (1 - stats.t.cdf(t_b1, gl_erro))*2
    
    if p_valor_b1 < alpha:
        b1_significante = "Sim"
        print(f"O coeficiente angular ({round(b1, 3)}) é diferente de zero (p-valor = {round(p_valor_b1, 3)})")
    else:
        b1_significante = "Não"
        print(f"O coeficiente angular ({round(b1, 3)}) é igual a zero (p-valor = {round(p_valor_b1, 3)})")
    
    # calculadno b0 interval
    b0_std = np.sqrt(MSQE*(1/df_data.shape[0] + df_data[x_name].mean()**2/Sxx))
    b0_ic = b0_std*t_critico
    
    
    # b0 significance test
    t_b0 = b0/b0_std
    p_valor_b0 = (1 - stats.t.cdf(t_b0, gl_erro))*2
    if p_valor_b0 < alpha:
        b0_significante = "Sim"
        print(f"O coeficiente linear ({round(b0, 3)}) é diferente de zero (p-valor = {round(p_valor_b0, 3)})")
    else:
        b0_significante = "Não"
        print(f"O coeficiente linear ({round(b0, 3)}) é igual a zero (p-valor = {round(p_valor_b1, 0)})")
    
    
    df_parameters = pd.DataFrame({
        "Parâmetros": ["Coeficiente angular", "Intercepto"],
        "Valor": [b1, b0],
        "Desvio padrão": [b1_std, b0_std],
        f"Intervalo confiança ({100*(1-alpha)}%)": [b1_ic, b0_ic],
        "Estatística": [t_b1, t_b0],
        "t-crítico": [t_critico, t_critico],
        "p-valor": [p_valor_b1, p_valor_b0],
        "Significativo?": [b1_significante, b0_significante]
    })
    
    
    return df_parameters, df_data


def anova(df_data, alpha=0.05):
    """calcula a ANOVA para regressão linear com intercepto.
    Df com: 
    - x_exp na primeira coluna;
    - y_exp_name na segunda coluna;
    - y_pred_name na terceira coluna;
    """
    
    # check if it is a df
    
    # sorting by concentration (first column)
    columns = df_data.columns
    x_exp_name, y_exp_name, y_pred_name = columns[0], columns[1], columns[2], 
    df_data = df_data.sort_values(by=[x_exp_name])
    
    # SQT
    df_data['sqt'] = np.square(df_data[y_exp_name] - df_data[y_exp_name].mean())
    SQT = df_data['sqt'].sum()
    gl_total = df_data.shape[0] - 1
    MSQT = SQT/gl_total
    
    # SQR
    df_data['sqr'] = np.square(df_data[y_pred_name] - df_data[y_exp_name].mean())
    SQR = df_data['sqr'].sum()
    gl_reg = 1
    MSQR = SQR/gl_reg
    
    # SQE
    df_data['sqe'] = np.square(df_data[y_exp_name] - df_data[y_pred_name])
    SQE = df_data['sqe'].sum()
    gl_erro = df_data.shape[0] - 2
    MSQE = SQE/gl_erro
    
    # statistic
    Fcalc = MSQR/MSQE
    
    # critical
    Ftab = stats.f.ppf(1-alpha, gl_reg, gl_erro)

    # p-value
    p_valor = 1- stats.f.cdf(Fcalc, gl_reg, gl_erro)

    # result
    if p_valor < alpha:
        anova_significativo = "Sim"
        print(f"A regressão é significativa.")
    else:
        anova_significativo = "Não"
        print(f"A regressão não é significativa.")

    
    df_anova = pd.DataFrame({
        "Fonte de Variação": ["Regressão", "Resíduos", "Total"],
        "Somatório": [SQR, SQE, SQT],
        "gl": [gl_reg, gl_erro, gl_total],
        "Médias": [MSQR, MSQE, MSQT],
        "F calculado": [Fcalc, "", ""],
        "F tabelado": [Ftab, "", ""],
        "p-valor": [p_valor, "", ""],
        "Singnificativo?": [anova_significativo, "", ""]
    })
    
    return df_data, df_anova


def calibration_interval(df_data, t_critico, MSQE):
    x_name = df_data.columns[0]
   
    Sxx = np.square(df_data[x_name] - df_data[x_name].mean()).sum()
    df_data["Médios IC"] = t_critico*np.sqrt(MSQE*(1/df_data.shape[0] + np.square(df_data[x_name] - df_data[x_name].mean())/Sxx))
    df_data["Médios L. Inferior"] = df_data['y_pred'] -  df_data["Médios IC"]
    df_data["Médios L. Superior"] = df_data['y_pred'] +  df_data["Médios IC"]

    df_data["Individuais IC"] = t_critico*np.sqrt(MSQE*(1 + 1/df_data.shape[0] + np.square(df_data[x_name] - df_data[x_name].mean())/Sxx))
    df_data["Individuais L. Inferior"] = df_data['y_pred'] -  df_data["Individuais IC"]
    df_data["Individuais L. Superior"] = df_data['y_pred'] +  df_data["Individuais IC"]
    
    return df_data