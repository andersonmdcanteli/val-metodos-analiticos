### Imports ###
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors
import seaborn as sns
import matplotlib as mpl






### Normality tests ###
def abdi_molin(x):
    """Esta função aplica o teste de Abdi-Molin para verificar a Normalidade dos dados
    """
    ABDIMOLIN_TABLE = {
    'n': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50),
    0.01 : (None, None, None, None, 0.4129, 0.3959, 0.3728, 0.3504, 0.3331, 0.3162, 0.3037, 0.2905, 0.2812, 0.2714, 0.2627, 0.2545, 0.2477,
                0.2408, 0.2345, 0.2285, 0.2226, 0.2190, 0.2141, 0.2090, 0.2053, 0.2010, 0.1985, 0.1941, 0.1911, 0.1886,
                0.1848, 0.1820, 0.1798, 0.1770, 0.1747, 0.1720, 0.1695, 0.1677, 0.1653, 0.1634, 0.1616, 0.1599, 0.1573,
                0.1556, 0.1542, 0.1525, 0.1512, 0.1499, 0.1476, 0.1463, 0.1457),
    0.05 : (None, None, None, None, 0.3754, 0.3427, 0.3245, 0.3041, 0.2875, 0.2744, 0.2616, 0.2506, 0.2426, 0.2337, 0.2257, 0.2196, 0.2128,
                0.2071, 0.2018, 0.1965, 0.1920, 0.1881, 0.1840, 0.1798, 0.1766, 0.1726, 0.1699, 0.1665, 0.1641, 0.1614,
                0.1590, 0.1559, 0.1542, 0.1518, 0.1497, 0.1478, 0.1454, 0.1436, 0.1421, 0.1402, 0.1386, 0.1373, 0.1353,
                0.1339, 0.1322, 0.1309, 0.1293, 0.1282, 0.1269, 0.1256, 0.1246),
    0.10 : (None, None, None, None, 0.3456, 0.3188, 0.2982, 0.2802, 0.2649, 0.2522, 0.2410, 0.2306, 0.2228, 0.2147, 0.2077, 0.2016, 0.1956,
                0.1902, 0.1852, 0.1803, 0.1764, 0.1726, 0.1690, 0.1650, 0.1619, 0.1589, 0.1562, 0.1533, 0.1509, 0.1483,
                0.1460, 0.1432, 0.1415, 0.1392, 0.1373, 0.1356, 0.1336, 0.1320, 0.1303, 0.1288, 0.1275, 0.1258, 0.1244,
                0.1228, 0.1216, 0.1204, 0.1189, 0.1180, 0.1165, 0.1153, 0.1142),
    }
    
    n_rep = x.size
    result = lilliefors(x, dist='norm', pvalmethod='table')
    if result[0] < ABDIMOLIN_TABLE[0.01][n_rep]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
        
    if result[0] < ABDIMOLIN_TABLE[0.05][n_rep]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False
        
    if result[0] < ABDIMOLIN_TABLE[0.10][n_rep]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False     
        
    return alfa_0_01, alfa_0_05, alfa_0_10



def anderson_darling(x):
    """Esta função aplica o teste de Anderson-Darling para verificar a Normalidade dos dados
    """
    statistic, critical_values, sig_level = stats.anderson(x)
    if statistic < critical_values[4]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
    if statistic < critical_values[2]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False

    if statistic < critical_values[1]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False    
        
    return alfa_0_01, alfa_0_05, alfa_0_10


def filliben(x):
    """Esta função aplica o teste de Filliben para verificar a Normalidade dos dados
    """
    FILLIBEN_TABLE = {
    'n': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50),
    0.01 : (None, None, None, 0.869, 0.822, 0.822, 0.835, 0.847, 0.859, 0.868, 0.876, 0.883, 0.889, 0.895, 0.901, 0.907, 0.912,
            0.916, 0.919, 0.923, 0.925, 0.928, 0.930, 0.933, 0.936, 0.937, 0.939, 0.941, 0.943, 0.945, 0.947, 0.948, 0.949,
            0.950, 0.951, 0.952, 0.953, 0.955, 0.956, 0.957, 0.958, 0.958, 0.959, 0.959, 0.960, 0.961, 0.962, 0.963, 0.963,
            0.964, 0.965),
    0.05 : (None, None, None, 0.879, 0.868, 0.879, 0.890, 0.899, 0.905, 0.912, 0.917, 0.922, 0.926, 0.931, 0.934, 0.937,
            0.940, 0.942, 0.945, 0.947, 0.950, 0.952, 0.954, 0.955, 0.957, 0.958, 0.959, 0.96, 0.962, 0.962, 0.964,
            0.965, 0.966, 0.967, 0.967, 0.968, 0.968, 0.969, 0.97, 0.971, 0.972, 0.972, 0.973, 0.973, 0.973, 0.974,
            0.974, 0.974, 0.975, 0.975, 0.977),
    0.10 : (None, None, None, 0.891, 0.894, 0.902, 0.911, 0.916, 0.924, 0.929, 0.934, 0.938, 0.941, 0.944, 0.947, 0.95, 0.952, 0.954,
            0.956, 0.958, 0.96, 0.961, 0.962, 0.964, 0.965, 0.966, 0.967, 0.968, 0.969, 0.969, 0.97, 0.971, 0.972, 0.973, 0.973,
            0.974, 0.974, 0.975, 0.975, 0.976, 0.977, 0.977, 0.978, 0.978, 0.978, 0.978, 0.979, 0.979, 0.98, 0.98, 0.981,),
    }
    
    n_rep = x.size
    x_y_data, params = stats.probplot(x)
    
    result = stats.pearsonr(x_y_data[0], x_y_data[1])

    if result[0] > FILLIBEN_TABLE[0.01][n_rep]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
        
    if result[0] > FILLIBEN_TABLE[0.05][n_rep]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False
        
    if result[0] > FILLIBEN_TABLE[0.10][n_rep]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False     

    return alfa_0_01, alfa_0_05, alfa_0_10    




def ks_test(x):
    """Esta função aplica o teste de Kolmogorov-Smirnov para verificar a Normalidade dos dados
    """
    KS_TABLE = {
        "n": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, None, None, None, None, 25, None, None, None, None, 30, None,
              None, None, None, 35),
        0.10: (None, 0.950, 0.776, 0.642, 0.564, 0.510, 0.470, 0.438, 0.411, 0.388, 0.368, 0.352, 0.338, 0.325, 0.314, 0.304, 0.295, 
               0.286, 0.278, 0.272, 0.264, None, None, None, None, 0.24, None, None, None, None, 0.22, None, None, None, None, 0.21),
        0.05: (None, 0.975, 0.842, 0.708, 0.624, 0.565, 0.521, 0.486, 0.457, 0.432, 0.410, 0.391, 0.375, 0.361, 0.349, 0.338, 0.328, 
               0.318, 0.309, 0.301, 0.294, None, None, None, None, 0.27, None, None, None, None, 0.24, None, None, None, None, 0.23),
        0.01: (None, 0.995, 0.929, 0.828, 0.733, 0.669, 0.618, 0.577, 0.543, 0.514, 0.490, 0.468, 0.450, 0.433, 0.418, 0.404, 0.392, 
               0.381, 0.371, 0.363, 0.356, None, None, None, None, 0.32, None, None, None, None, 0.29, None, None, None, None, 0.27)
        }    
    n_rep = x.size
    result = stats.kstest(x, cdf='norm', args=(x.mean(), x.std(ddof=1)), N=n_rep)
    if result[0] < KS_TABLE[0.01][n_rep]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
        
    if result[0] < KS_TABLE[0.05][n_rep]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False
        
    if result[0] < KS_TABLE[0.1][n_rep]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False     
        
    return alfa_0_01, alfa_0_05, alfa_0_10



def lilliefors_test(x):
    """Esta função aplica o teste de Lilliefors para verificar a Normalidade dos dados
    """
    LILLIEFORS_TABLE = {
        'n': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
        0.01 : (None, None, None, None, 0.417, 0.405, 0.364, 0.348, 0.331, 0.311, 0.294, 0.284, 0.275, 0.268, 0.261, 0.257, 0.250, 0.245, 0.239,
                    0.235, 0.231, None, None, None, None, 0.203, None, None, None, None, 0.187),
        0.05 : (None, None, None, None, 0.381, 0.337, 0.319, 0.300, 0.285, 0.271, 0.258, 0.249, 0.242, 0.234, 0.227, 0.220, 0.213, 0.206, 0.200,
                    0.195, 0.190, None, None, None, None, 0.180, None, None, None, None, 0.161),
        0.10 : (None, None, None, None, 0.352,  0.315, 0.294, 0.276, 0.261, 0.249, 0.239, 0.230, 0.223, 0.214, 0.207, 0.201, 0.195, 0.189, 0.184,
                    0.179, 0.174, None, None, None, None, 0.165, None, None, None, None, 0.144)
        }  
    n_rep = x.size
    result = lilliefors(x, dist='norm', pvalmethod='table')
    if result[0] < LILLIEFORS_TABLE[0.01][n_rep]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
        
    if result[0] < LILLIEFORS_TABLE[0.05][n_rep]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False
        
    if result[0] < LILLIEFORS_TABLE[0.1][n_rep]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False     
        
    return alfa_0_01, alfa_0_05, alfa_0_10


def shapiro_wilk_test(x):  
    """Esta função aplica o teste de Shapiro-Wilk para verificar a Normalidade dos dados
    """
    shapiro_wilk_table = {
        "n": (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50),
        0.01 : (None, None, None, 0.753, 0.687, 0.686, 0.713, 0.730, 0.749, 0.764, 0.781, 0.792, 0.805, 0.814, 0.825, 0.835, 0.844, 0.851, 
                0.858, 0.863, 0.868, 0.873, 0.878, 0.881, 0.884, 0.888, 0.891, 0.894, 0.896, 0.898, 0.900, 0.902, 0.904, 0.906, 0.908, 
                0.910, 0.912, 0.914, 0.916, 0.917, 0.919, 0.920, 0.922, 0.923, 0.924, 0.926, 0.927, 0.928, 0.929, 0.929, 0.930),
        0.05 : (None, None, None, 0.767, 0.748, 0.762, 0.788, 0.803, 0.818, 0.829, 0.842, 0.850, 0.859, 0.866, 0.874, 0.881, 0.887, 0.892, 
                0.897, 0.901, 0.905, 0.908, 0.911, 0.914, 0.916, 0.918, 0.920, 0.923, 0.924, 0.926, 0.927, 0.929, 0.930, 0.931, 0.933, 
                0.934, 0.935, 0.936, 0.938, 0.939, 0.940, 0.941, 0.942, 0.943, 0.944, 0.945, 0.945, 0.946, 0.947, 0.947, 0.947),
        0.10 : (None, None, None, 0.789, 0.792, 0.806, 0.826, 0.838, 0.851, 0.859, 0.869, 0.876, 0.883, 0.889, 0.895, 0.901, 0.906, 0.910, 
                0.914, 0.917, 0.920, 0.923, 0.926, 0.928, 0.930, 0.931, 0.933, 0.935, 0.936, 0.937, 0.939, 0.940, 0.941, 0.942, 0.943, 
                0.944, 0.945, 0.946, 0.947, 0.948, 0.949, 0.950, 0.951, 0.951, 0.952, 0.953, 0.953, 0.954, 0.954, 0.955, 0.955),
    }
    
    n_rep = x.size
    result = stats.shapiro(x)
       
    if result[0] > shapiro_wilk_table[0.01][n_rep]:
        alfa_0_01 = True
    else:
        alfa_0_01 = False
        
    if result[0] > shapiro_wilk_table[0.05][n_rep]:
        alfa_0_05 = True
    else:
        alfa_0_05 = False     
        
    if result[0] > shapiro_wilk_table[0.1][n_rep]:
        alfa_0_10 = True
    else:
        alfa_0_10 = False     
        
    return alfa_0_01, alfa_0_05, alfa_0_10





### Plots ###
def make_heatmap(df_data, title, n_samples, normal=True):
    
    if normal:
        true = 1
        false = 0
    else:
        true = 0
        false = 1
    
    df = df_data.copy()
    df["Abdi-Molin"] = df["Abdi-Molin"].replace({True: true, False: false})
    df["Anderson-Darling"] = df["Anderson-Darling"].replace({True: true, False: false})
    df["Filliben"] = df["Filliben"].replace({True: true, False: false})
    df["Kolmogorov-Smirnov"] = df["Kolmogorov-Smirnov"].replace({True: true, False: false})
    df["Lilliefors"] = df["Lilliefors"].replace({True: true, False: false})
    df["Shapiro-Wilk"] = df["Shapiro-Wilk"].replace({True: true, False: false})
    
    df_10 = df[df["Alpha"] == "0.10"]
    df_05 = df[df["Alpha"] == "0.05"]
    df_01 = df[df["Alpha"] == "0.01"]
    df_10 = df_10.drop("Alpha", axis=1)
    
    df_10 = df_10.rename(columns={
        "Abdi-Molin": "Abdi-Molin (10%)",
        "Anderson-Darling": "Anderson-Darling (10%)",
        "Filliben": "Filliben (10%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (10%)",
        "Lilliefors": "Lilliefors (10%)",
        "Shapiro-Wilk": "Shapiro-Wilk (10%)",
    })
    df_05 = df_05.drop("Alpha", axis=1)
    df_05 = df_05.rename(columns={
        "Abdi-Molin": "Abdi-Molin (5%)",
        "Anderson-Darling": "Anderson-Darling (5%)",
        "Filliben": "Filliben (5%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (5%)",
        "Lilliefors": "Lilliefors (5%)",
        "Shapiro-Wilk": "Shapiro-Wilk (5%)",
    })
    df_01 = df_01.drop("Alpha", axis=1)
    df_01 = df_01.rename(columns={
        "Abdi-Molin": "Abdi-Molin (1%)",
        "Anderson-Darling": "Anderson-Darling (1%)",
        "Filliben": "Filliben (1%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (1%)",
        "Lilliefors": "Lilliefors (1%)",
        "Shapiro-Wilk": "Shapiro-Wilk (1%)",
    })
    
    df_10 = df_10[
        ["Abdi-Molin (10%)", "Anderson-Darling (10%)", "Filliben (10%)", "Kolmogorov-Smirnov (10%)", "Lilliefors (10%)", "Shapiro-Wilk (10%)"]
    ].groupby(df_10["n amostral"]).agg("sum")
    df_05 = df_05[
        ["Abdi-Molin (5%)", "Anderson-Darling (5%)", "Filliben (5%)", "Kolmogorov-Smirnov (5%)", "Lilliefors (5%)", "Shapiro-Wilk (5%)"]
    ].groupby(df_05["n amostral"]).agg("sum")
    df_01 = df_01[
        ["Abdi-Molin (1%)", "Anderson-Darling (1%)", "Filliben (1%)", "Kolmogorov-Smirnov (1%)", "Lilliefors (1%)", "Shapiro-Wilk (1%)"]
    ].groupby(df_01["n amostral"]).agg("sum")
    
    df = pd.concat([df_10, df_05, df_01], axis='columns')

    df = df*100/n_samples
    fig, ax = plt.subplots(figsize=(14,6))
    ax = sns.heatmap(df.transpose(), annot=True, ax=ax, fmt=".1f", cmap=sns.color_palette("flare", as_cmap=True), vmin=0, vmax=100)
    ax.set_title(title)
    plt.show()    

def make_bar_plot(df_data, n_samples, title, kind=True):
    df = df_data.copy()
    
    COLOR_10 = "#7E9DC2"
    COLOR_5 = "#F8CF2C"
    COLOR_1 = "#DB55D4"
    

    if kind:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 1, False: 0})
        df_agg = df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = df_agg[df_agg.columns[0]].unstack().plot(kind='bar', ax=ax, color=[COLOR_1, COLOR_5, COLOR_10])
        ax.set_ylabel("Porcentagem de acertos (%)")
        ax.axhline(y=1, color=COLOR_1, ls="--")
        ax.axhline(y=5, color=COLOR_5, ls="--")
        ax.axhline(y=10, color=COLOR_10, ls="--")        
    else:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 0, False: 1})
        df_agg = df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = df_agg[df_agg.columns[0]].unstack().plot(kind='bar', ax=ax, color=[COLOR_1, COLOR_5, COLOR_10])
        ax.set_ylabel("Power (%)")       
        ax.axhline(y=90, color="gray", ls="--")
        ax.axhline(y=80, color="gray", ls="--")
        ax.axhline(y=70, color="gray", ls="--")
        
    ax.set_xlabel("Número de amostras")
    ax.tick_params(axis='x', rotation=0)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.115, 0.95), fancybox=True, shadow=True, title="Alpha",)
    fig.tight_layout()
    plt.show()


    
def plot_all_results(df_data):
    df = df_data.copy()
    df.insert(0, 'Distribution', df.pop('Distribution'))
    arr = df.sum(numeric_only = True)
    arr.sort_values(inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = arr.plot(kind="bar", ax=ax, color="#C0C0C0")
    ax.set_ylabel("Total de acertos")
    plt.show()
    
    
def plot_all_results_percentage(df_data, n_samples, kind=True):
    df = df_data.copy()
    df.insert(0, 'Distribution', df.pop('Distribution'))
    arr = df.sum(numeric_only = True)*100/(df.shape[0]*n_samples)
    arr.sort_values(inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = arr.plot(kind="bar", ax=ax, color="#C0C0C0")
    ax.set_ylabel("Porcentagem de acertos (%)")
    if kind == False:
        ax.set_ylabel("Power (%)")       
        ax.axhline(y=90, color="gray", ls="--")
        ax.axhline(y=80, color="gray", ls="--")
        ax.axhline(y=70, color="gray", ls="--")
    plt.show()
    
    
def make_kernal_plot(x, title):
    fig, ax = plt.subplots(1,4, figsize=(14,3))
    sns.kdeplot(x, label="Dataset", color="blue", ax=ax[0])
    sns.kdeplot(x[:5], label="n=5", color="red",  ax=ax[0])
    ax[0].legend(fontsize=8)

    sns.kdeplot(x, label="Dataset", color="blue", ax=ax[1])
    sns.kdeplot(x[:10], label="n=10", color="red",  ax=ax[1])
    ax[1].legend(fontsize=8)

    sns.kdeplot(x, label="Dataset", color="blue", ax=ax[2])
    sns.kdeplot(x[:15], label="n=15", color="red",  ax=ax[2])
    ax[2].legend(fontsize=8)

    sns.kdeplot(x, label="Dataset", color="blue", ax=ax[3])
    sns.kdeplot(x[:25], label="n=25", color="red",  ax=ax[3])
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.88)

    plt.show()  
    
    
def make_bar_plot_average_alpha(df_data, n_samples, title, kind=True):
    df = df_data.copy()


    if kind:
        
        df[df.columns[0]] = df[df.columns[0]].replace({True: 1, False: 0})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples        
        
        df_agg = df_agg[df_agg.columns[0]].unstack()
        ylabel = "Average of correct (%)"
    else:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 0, False: 1})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples            
        
        df_agg = df_agg[df_agg.columns[0]].unstack()
        ylabel = "Mean Power (%)"

        
    df_agg["mean"] = df_agg.mean(axis=1)
    plt.figure(figsize=(10,4))
    plt.bar(df_agg.index.astype(str), df_agg["mean"], color="#C0C0C0")
    if kind == False:
        plt.axhline(y=90, color="gray", ls="--")
        plt.axhline(y=80, color="gray", ls="--")
        plt.axhline(y=70, color="gray", ls="--")
    plt.xlabel("Sample size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
    
    
    
def make_global_average_bar_plot(dfs, names, n_samples, title, kind=True):
    results = []
    for df, name in zip(dfs, names):
        results.append(make_calc_average_alpha(df, n_samples, name, kind=kind))
    df = pd.concat(results)
    
    fig, ax = plt.subplots(figsize=(10,4))

    ax = sns.barplot(data=df, x=df.index, y='mean', hue='Test', ax=ax, palette="Paired")
    if kind == False:
        ax.axhline(y=90, color="gray", ls="--")
        ax.axhline(y=80, color="gray", ls="--")
        ax.axhline(y=70, color="gray", ls="--")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Power average (%)")
    ax.set_title(title)
    plt.show()
    
    
    
    
def make_plot_average_alpha_size(dfs, names, n_samples, title, kind=True):
    results = []
    for df, name in zip(dfs, names):
        results.append(make_calc_average_alpha_size(df, n_samples, name, kind=kind))
    df = pd.concat(results)
    fig, ax = plt.subplots(figsize=(10,4))

    ax = sns.barplot(data=df, x="Test", y='Full Mean', ax=ax, palette="Paired")
    ax.set_ylabel("Global mean Power (%)")
    ax.set_xlabel(None)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    plt.show()
    
    
    
    
def make_skew_kurtosis_plot(df, seeds, data_kurtosis, data_skew, suptitle, kind=True):
    alpha = ["0.01", "0.05", "0.10"]
    dfs = []
    for i in range(3):
        df_data = df[df["Alpha"] == alpha[i]].copy()
        count = []

        for seed in seeds:
            df_aux = df_data[df_data["Seed"] == seed].copy()
            df_aux.reset_index(inplace=True, drop=True)
            if kind:
                df_aux["Resultado"] = df_aux["Resultado"].replace({True: 1, False: 0})
            else:
                df_aux["Resultado"] = df_aux["Resultado"].replace({True: 0, False: 1})
            count.append(df_aux["Resultado"].sum())

        dfs.append(pd.DataFrame({
            "Seed": seeds,
            "Count": count,
            "Kurtosis": data_kurtosis,
            "Skewness": data_skew,
            "Alpha": [alpha[i]]*len(seeds)
        }))
        
    df_result = pd.concat(dfs)
    df_result.merge(df)
        
    palette = "vlag_r"
    g = sns.FacetGrid(df_result, col='Alpha', palette = palette, height=4)

    def facet_scatter(x, y, c, **kwargs):
        """Draw scatterplot with point colors from a faceted DataFrame columns."""
        kwargs.pop("color")
        plt.scatter(x, y, c=c, **kwargs)

    vmin, vmax = 1, 19
    cmap = sns.diverging_palette(240, 10, l=65, center="dark", as_cmap=True)

    g = g.map(facet_scatter, 'Skewness', 'Kurtosis', "Count",
              s=50, alpha=1, vmin=vmin, vmax=vmax, cmap=palette, edgecolors="gray", linewidths=.5)

    # Make space for the colorbar
    g.fig.subplots_adjust(right=.92)

    # Define a new Axes where the colorbar will go
    cax = g.fig.add_axes([.94, .25, .02, .6])

    # Get a mappable object with the same colormap as the data
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=palette)
    g.figure.suptitle(suptitle, size=16, y=1.05)
    g.figure.subplots_adjust(top=.9)

    # Draw the colorbar
    g.figure.colorbar(points, cax=cax)

    plt.show()
    
    
    
    
def power_summing_bar(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(data=df, x=df.index, y=df.columns[0], hue=df.columns[1], ax=ax, palette="Paired")
    plt.show()
    
    
    
    
def display_distribution_plots(df, data_skew, data_kurtosis, seeds, distribution):
    
    fig, ax = plt.subplots(1, 3, figsize=(14,4))
    ax[0].boxplot([data_skew, data_kurtosis, ], labels=["Kurtosis", "Skewness"], widths=.4, medianprops=dict(color="k"))
    ax[1].scatter(data_skew, data_kurtosis, alpha=.1, c="k")
    ax[1].set_xlabel("Skewness")
    ax[1].set_ylabel("Kurtosis")
    for seed in seeds:
        data = df[df["seed"] == seed]["Data"].copy()
        data = data.sort_values()
        if distribution == "norm":
            x = np.linspace(stats.norm.ppf(0.0001, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.norm.ppf(0.9999, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.norm.pdf(x, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Normal distribution"
        elif distribution == "cauchy":
            x = np.linspace(stats.cauchy.ppf(0.01, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.cauchy.ppf(0.99, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.cauchy.pdf(x, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Cauchy distribution"   
            
        elif distribution == "chisquare-df1":
            x = np.linspace(stats.chi2.ppf(0.01, df=1, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.chi2.ppf(0.99, df=1, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.chi2.pdf(x, df=1, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Chi-square distribution (df=1)"   
            
        elif distribution == "chisquare-df5":
            x = np.linspace(stats.chi2.ppf(0.01, df=5, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.chi2.ppf(0.99, df=5, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.chi2.pdf(x, df=1, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Chi-square distribution (df=5)"   
            
        elif distribution == "chisquare-df20":
            x = np.linspace(stats.chi2.ppf(0.01, df=20, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.chi2.ppf(0.99, df=20, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.chi2.pdf(x, df=1, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Chi-square distribution (df=20)"   
            
        elif distribution == "exponential":
            x = np.linspace(stats.expon.ppf(0.01, loc=data.mean(), scale=data.std(ddof=1)), 
                            stats.expon.ppf(0.99, loc=data.mean(), scale=data.std(ddof=1)), 1000)
            ax[2].plot(x, stats.expon.pdf(x, loc=data.mean(), scale=data.std(ddof=1)),  c="k", alpha=.1)
            # ax[2].set_xlim([x[0], x[-1]])
            title = "Exponential distribution"   
        
        else:
            raise ValueError
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Density")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
### Calcs ###
def make_calc(df_data, normal=True):
    
    if normal:
        true = 1
        false = 0
    else:
        true = 0
        false = 1
    
    df = df_data.copy()
    df["Abdi-Molin"] = df["Abdi-Molin"].replace({True: true, False: false})
    df["Anderson-Darling"] = df["Anderson-Darling"].replace({True: true, False: false})
    df["Filliben"] = df["Filliben"].replace({True: true, False: false})
    df["Kolmogorov-Smirnov"] = df["Kolmogorov-Smirnov"].replace({True: true, False: false})
    df["Lilliefors"] = df["Lilliefors"].replace({True: true, False: false})
    df["Shapiro-Wilk"] = df["Shapiro-Wilk"].replace({True: true, False: false})
    df_10 = df[df["Alpha"] == "0.10"]
    df_05 = df[df["Alpha"] == "0.05"]
    df_01 = df[df["Alpha"] == "0.01"]
    df_10 = df_10.drop("Alpha", axis=1)
    df_10 = df_10.rename(columns={
        "Abdi-Molin": "Abdi-Molin (10%)",
        "Anderson-Darling": "Anderson-Darling (10%)",
        "Filliben": "Filliben (10%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (10%)",
        "Lilliefors": "Lilliefors (10%)",
        "Shapiro-Wilk": "Shapiro-Wilk (10%)",
    })
    df_05 = df_05.drop("Alpha", axis=1)
    df_05 = df_05.rename(columns={
        "Abdi-Molin": "Abdi-Molin (5%)",
        "Anderson-Darling": "Anderson-Darling (5%)",
        "Filliben": "Filliben (5%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (5%)",
        "Lilliefors": "Lilliefors (5%)",
        "Shapiro-Wilk": "Shapiro-Wilk (5%)",
    })
    df_01 = df_01.drop("Alpha", axis=1)
    df_01 = df_01.rename(columns={
        "Abdi-Molin": "Abdi-Molin (1%)",
        "Anderson-Darling": "Anderson-Darling (1%)",
        "Filliben": "Filliben (1%)", 
        "Kolmogorov-Smirnov": "Kolmogorov-Smirnov (1%)",
        "Lilliefors": "Lilliefors (1%)",
        "Shapiro-Wilk": "Shapiro-Wilk (1%)",
    })
    
    df_10 = df_10[
        ["Abdi-Molin (10%)", "Anderson-Darling (10%)", "Filliben (10%)", "Kolmogorov-Smirnov (10%)", "Lilliefors (10%)", "Shapiro-Wilk (10%)"]
    ].groupby(df_10["n amostral"]).agg("sum")
    df_05 = df_05[
        ["Abdi-Molin (5%)", "Anderson-Darling (5%)", "Filliben (5%)", "Kolmogorov-Smirnov (5%)", "Lilliefors (5%)", "Shapiro-Wilk (5%)"]
    ].groupby(df_05["n amostral"]).agg("sum")
    df_01 = df_01[
        ["Abdi-Molin (1%)", "Anderson-Darling (1%)", "Filliben (1%)", "Kolmogorov-Smirnov (1%)", "Lilliefors (1%)", "Shapiro-Wilk (1%)"]
    ].groupby(df_01["n amostral"]).agg("sum")
    
    df = pd.concat([df_10, df_05, df_01], axis='columns')    
    return df
    
    
    
def flatten(l):
    return [item for sublist in l for item in sublist]
    
    
def make_calc_average_alpha(dfs, n_samples, title, kind=True):
    df = dfs.copy()

    if kind:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 1, False: 0})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")

        df_agg = df_agg*100/n_samples
        df_agg = df_agg[df_agg.columns[0]].unstack()
    else:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 0, False: 1})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")

        df_agg = df_agg*100/n_samples
        df_agg = df_agg[df_agg.columns[0]].unstack()
    df_agg["mean"] = df_agg.mean(axis=1)
    df_agg["Test"] = title
    return df_agg
    
    
    
    
def make_calc_average_alpha_size(dfs, n_samples, title, kind=True):
    df = dfs.copy()

    if kind:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 1, False: 0})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples        
        df_agg = df_agg[df_agg.columns[0]].unstack()
    else:
        df[df.columns[0]] = df[df.columns[0]].replace({True: 0, False: 1})
        df_agg =  df.groupby([df.columns[1], df.columns[2]]).agg("sum")
        df_agg = df_agg*100/n_samples    
        
        df_agg = df_agg[df_agg.columns[0]].unstack()

        
    df_agg["mean"] = df_agg.mean(axis=1)
    df_agg["Test"] = title
    df_result = pd.DataFrame({
        "Full Mean": [df_agg["mean"].mean()],
        "Test": [title]
    })
    return df_result
    
    
    
    
    
    
    
def ranking(dfs, names, n_samples, kind=True):
    results = []
    for df, name in zip(dfs, names):
        results.append(make_calc_average_alpha_size(df, n_samples, name, kind=kind))
    df = pd.concat(results)
    df.reset_index(inplace=True, drop=True)
    df["rank"] = df["Full Mean"].rank(ascending=False)
    
    return df
    
    
    
    
    
def calc_power_summing(df, power, test_name, n_samples):

    def power_checker(x, power):
        if x < power:
            return 0
        else:
            return 1

    df[df.columns[0]] = df[df.columns[0]].replace({True: 0, False: 1})
    df_agg = df.groupby([df.columns[1], df.columns[2]]).agg("sum")
    df_agg = df_agg*100/n_samples
    power_name = f"Power Result (beta={round(1-power/100, 2)})"
    df_agg[power_name] = df_agg["Resultado"].apply(power_checker, args=(power,))
    df_agg = df_agg[power_name].groupby(level=[0]).sum().to_frame()
    df_agg["Test"] = test_name
    return df_agg 
    
    
    
    
def display_skew_kurt_info(data_skew, data_kurtosis):
    data_skew = np.array(data_skew)
    print(f"""Skewness: 
            mean={round(data_skew.mean(),3)}
            std={round(data_skew.std(ddof=1),3)}
            min={round(data_skew.min(),3)}
            max={round(data_skew.max(),3)}"""
    )
    data_kurtosis = np.array(data_kurtosis)
    print(f"""Kurtosis: 
            mean={round(data_kurtosis.mean(),3)}
            std={round(data_kurtosis.std(ddof=1),3)}
            min={round(data_kurtosis.min(),3)} 
            max={round(data_kurtosis.max(),3)}"""
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#