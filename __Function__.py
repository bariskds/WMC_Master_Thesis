import numpy as np 
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import kurtosis, skew, mode

import plotly.graph_objects as go


class WMC(object):
    '''WMV(g,C)
    
        g is a N x V matrix of derivatives payouts
        C is a 1 x N matrix of derivative prices
        
        N = number of instruments
        V = number of paths
        
        Once initiated call .solve to calculate optimal lambdas
        Resulting probabilities are stored in variable p
    '''

    def __init__(self, g, C):
        self.g = np.array(g)   # N_paths x N_options
        self.C = np.array(C)   # N_options
        self.v = self.g.shape[0]
        self.last = None
        
    def recalc(self, lambda_):
        if self.last != tuple(lambda_.tolist()):
            lambda_ = np.matrix(lambda_)
            self.egl = np.exp(self.g * lambda_.T)
            self.Z = np.sum(self.egl)
            self.p = (self.egl / self.Z).T
            self.w = float(np.log(self.Z) - self.C * lambda_.T + self.e/2 * (lambda_*lambda_.T))
            self.fPrime = np.array((self.p * self.g - self.C + self.e * lambda_))[0]
            self.last = tuple(lambda_.tolist())
            
    def _objective(self, lambda_):
        self.recalc(lambda_)
        return self.w
        
    def _fPrime(self, lambda_):
        self.recalc(lambda_)
        return self.fPrime
    
    def solve(self, fPrime=True, e=1e-5, method="L-BFGS-B", bounds=None, options=None):
        ''' fPrime = True/False (use gradiant)
            e      = "weight" in the least squares implementation
            disp   = Print optimization convergence
        '''
        self.e=e
        x0 = np.zeros_like(self.C)

        if fPrime:
            result = minimize(self._objective, x0, jac=self._fPrime, method=method, bounds=bounds, options=options)
        else:
            result = minimize(self._objective, x0, method=method, bounds=bounds, options=options)
        
        self.opt_result = result
        self.lambda_ = np.array(result.x)  # Store lambdas
        
        return self.p
    

class Comparaison():
    
    def saving_data(type_error, name_save):

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in type_error.columns],
                fill_color="#2e6b57",
                font=dict(color='white', size=12),
                align='center',
                height=40),
                
                cells=dict(
                    values=[type_error[type_error.columns[0]], type_error[type_error.columns[1]]],
                    fill_color='#f5f5fa',
                    align='center',
                    font_size=10,
                    height=30)
                    )])

        fig.update_layout(width=2500, height=500, margin=dict(l=0, r=0, t=0, b=0))
        fig.write_image(fr"C:\Users\baris\Desktop\Master Thesis\2. Writing\{name_save}.png", scale=3)

    def comparaison_maturity(data):

        compared_maturity = data.groupby(by=['Maturity'])[['Price','Error']].mean().reset_index()

        list_error_maturity = []
        list_error_maturity.append(f"{compared_maturity[compared_maturity['Maturity'] < 0.3]['Error'].mean():.4f}")
        list_error_maturity.append(f"{compared_maturity[(compared_maturity['Maturity'] >= 0.3) & (compared_maturity['Maturity'] < 0.6)]['Error'].mean():.4f}")
        list_error_maturity.append(f"{compared_maturity[(compared_maturity['Maturity'] >= 0.6)]['Error'].mean():.4f}")

        list_mean_maturity = []
        list_mean_maturity.append(f"{compared_maturity[compared_maturity['Maturity'] < 0.3]['Price'].mean():.4f}")
        list_mean_maturity.append(f"{compared_maturity[(compared_maturity['Maturity'] >= 0.3) & (compared_maturity['Maturity'] < 0.6)]['Price'].mean():.4f}")
        list_mean_maturity.append(f"{compared_maturity[(compared_maturity['Maturity'] >= 0.6)]['Price'].mean():.4f}")

        list_com_maturity = []
        list_com_maturity.append(f'Mean (Filtered for maturity < 0.3, Price = {list_mean_maturity[0]})')
        list_com_maturity.append(f'Mean (Filtered for maturity >= 0.3 and < 0.6, Price = {list_mean_maturity[1]})')
        list_com_maturity.append(f'Mean (Filtered for maturity >= 0.6, Price = {list_mean_maturity[2]})')

        error_metrics_maturity = pd.DataFrame(data={"Metric Maturity": list_com_maturity, "Mean Eror (abs)": list_error_maturity})

        return error_metrics_maturity
    
    def comparaison_strike(data, price):

        compared_strike = data.groupby(by=['Strike'])[['Price','Error']].mean().reset_index()
        compared_strike['Strike'] = compared_strike['Strike'].map(lambda x: x/price*100)

        list_error_strike = []
        list_error_strike.append(f"{compared_strike[compared_strike['Strike'] < 100]['Error'].mean():.4f}")
        list_error_strike.append(f"{compared_strike[(compared_strike['Strike'] >= 100) & (compared_strike['Strike'] < 105)]['Error'].mean():.4f}")
        list_error_strike.append(f"{compared_strike[(compared_strike['Strike'] >= 105)]['Error'].mean():.4f}")

        list_mean_strike = []
        list_mean_strike.append(f"{compared_strike[compared_strike['Strike'] < 100]['Price'].mean():.4f}")
        list_mean_strike.append(f"{compared_strike[(compared_strike['Strike'] >= 100) & (compared_strike['Strike'] < 105)]['Price'].mean():.4f}")
        list_mean_strike.append(f"{compared_strike[(compared_strike['Strike'] >= 105)]['Price'].mean():.4f}")

        list_com_strike = []
        list_com_strike.append(f'Mean (Filtered for strike < 100, Price = {list_mean_strike[0]})')
        list_com_strike.append(f'Mean (Filtered for strike >= 100 and < 105, Price = {list_mean_strike[1]})')
        list_com_strike.append(f'Mean (Filtered for strike >= 105, Price = {list_mean_strike[2]})')

        error_metrics_strike = pd.DataFrame(data={"Metric Strike": list_com_strike, "Mean Eror (abs)": list_error_strike})

        return error_metrics_strike
    
    def comparaison_surface(data):
    
        list_quantile = [0.2, 0.5, 0.8]

        list_error_mean = []
        list_com = []

        list_error_mean.append(f"{data[data['Price'] < data['Price'].quantile(0.2)]['Error'].mean():.4f}")
        list_com.append(f"Mean (Filtered for price < {data['Price'].quantile(0.2):.2f}$, 20% Quantile)")

        for quantile in list_quantile:
            list_error_mean.append(f"{data[data['Price'] > data['Price'].quantile(quantile)]['Error'].mean():.4f}")
            list_com.append(f"Mean (Filtered for price > {data['Price'].quantile(quantile):.2f}$, {quantile*100:.0f}% Quantile)")

        list_error_mean.append(f"{data['Error'].mean():.4f}")
        list_com.append(f"Mean")

        list_error_mean.append(f"{data['Error'].median():.4f}")
        list_com.append(f"Median")

        error_metrics_surface = pd.DataFrame(data={"Metric Surface": list_com,"Mean Eror (abs)": list_error_mean})

        return error_metrics_surface
    
    def stats_weights(weights):

        N = len(weights)
        mean = np.mean(weights)
        stderr = np.std(weights) / np.sqrt(N)
        median = np.median(weights)
        mode_val = mode(weights, keepdims=True)[0][0]
        std = np.std(weights)
        var = np.var(weights)
        kurt = kurtosis(weights, fisher=False)
        skewness = skew(weights)
        _range = np.ptp(weights)
        min_val = np.min(weights)
        max_val = np.max(weights)

        df_stats = pd.DataFrame({"Statistic": [
            "Mean", "Standard Error", "Median", "Mode", "Standard Deviation",
            "Sample Variance", "Kurtosis", "Skewness", "Range", "Minimum", "Maximum"],
            "Value": [mean, stderr, median, mode_val, std, var, kurt, skewness, _range, min_val, max_val]
            })

        df_stats['Value'] = df_stats['Value'].apply(lambda x: f"{x:.6g}")
        
        return df_stats