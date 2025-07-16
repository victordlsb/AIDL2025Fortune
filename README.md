# FORTUNE: Financial Outlook Realized by Unified Network Encoder

**Disclaimer: no fortune will be found here**

Link to the slides: 

https://docs.google.com/presentation/d/1YvCqIjJiYBOI3JZL2JlWEd81KjGHN89CCODXsdoQKiI/edit?usp=sharing

# Introduction

Beating the stock market is a wish that any investor has dreamed of and very few have been able. There are several ways of trading, but when it comes to stock trading, strategies can be divided between active and passive. 

**Active strategies** involve frequent buying and selling of securities with the goal of outperforming the market or a specific benchmark. These strategies rely on market analysis, timing, and the skill of the investor or fund manager to identify short-term opportunities. Day trading based on technical analysis is a classic example of an active strategy. 

![Some examples of stock patterns used in technical analysis](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/81d49b50-c112-4065-ace6-0948237a93ca.png)

Some examples of stock patterns used in technical analysis

In contrast, passive strategies aim to match the performance of a market index rather than beat it. Investors typically buy and hold a diversified portfolio over the long term, such as through index funds or Exchange Traded Funds (ETFs). Passive investing involves minimal trading and usually comes with lower fees, making it a popular choice for long-term wealth building. 

![S&P 500 returns in 30 years](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image.png)

S&P 500 returns in 30 years

It is well known that passive investors generally have more consistent success than active investors. The S&P 500 index has an inflation-adjusted return of around 7%, with the nominal being around 10% before inflation. 

Technical analysis is highly critisiced due to its lack of scientific foundation as it is based in subjective pattern recognition, the disregard to underlying company and economic data or how prone to bias it is. Yet some people still use active strategies (and even beat passive investment with it!). 

So… what if we had an AI trained to do what technical analysis day traders do, but with the power to process data in an exponentially higher volume and speed? Since some people make money recognising patterns… can an AI also make money with it? Can we beat the S&P 500 during one year of trading? Let’s find out!

# The data

Since we are going to predict stock price movements, we need stock data. In technical analysis, for a single stock there are 5 features given for an interval of time where there has been trading:

- Open: The price at the beginning of the time step
- High: The highest price reached by the stock during the time step
- Close: The price at the end of the time step
- Low: The lowest price reached by the stock during the time step
- Volume: How many stocks were traded during the time period

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%201.png)

These are usually displayed using candlestick charts, where each candlestick represents the time step. The timestep can be from 1 minute to days, months or even years. The resolution really depends on what you are analysing. For day trading, 1 minute or 5 minute timesteps are used. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%202.png)

We decided to use the data from the 503 stocks that are in the S&P 500 (503 stocks traded by 500 companies), with a resolution of 1 minute, since we want to day trade. NOTE: The amount of stocks were reduced as it will be explained later. We obtained the stocks from [EODHD](eodhd.com), which had 1 minute data starting 2004 (and it was the cheapest option we found thanks to its student discount rate).

Use the [EODHD.py](http://EODHD.py) file to download stock data. Here there is an example on how to download the data for Apple, Tesla and Microsoft

```python
from EODHD import download_stock
import concurrent.futures

API_KEY = "" #Insert your EODHD API Key
data_folder = './EODHD_Data' # Path to your data folder
tickers_to_download = ['AAPL','TSLA','MSFT'] #List of stocks to download
initial_date_utc = int(datetime.strptime("2004-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
final_date_utc = int(datetime.strptime("2025-05-23 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
time_interval = 120*24*60*60 # In seconds. This equivalents to 120 days, max allowed by EODHD
reverse = True # Set to true to start backwards from newest to oldest, useful since some stocks might be newer than your initial date
max_workers = 20 # Parallel execution,set 1 to prevent parallelization
max_missed_steps = 5 #set how many time intervals can return no data before stopping the stock download e.g: 5 means 5 intervals of 120 days. 

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(download_stock, stock_symbol, idx, len(tickers_to_download), initial_date_utc, final_date_utc, reverse=False, max_missed_steps = 5)
        for idx, (stock_symbol) in enumerate(tickers_to_download)
    ]
    concurrent.futures.wait(futures)
```

# Preprocessing the data

The 503 stocks data dating from 1st of January of 2004 until the 23rd of May of 2025 with 1 minute intervals summed more than 45 gigabytes of data! CSV is not precisely the most efficient file format, specially for very large sets of data. HDF5 comes to the rescue. We transform all the csv files into HDF5 files with the code from csv_to_hdf5.py. We use this also to align all our files with our desired dates, starting in 2004-01-01 00:00:00 UTC Time. If data is missing for a given timestep, it will be filled with NaN which we will mask later in our code. 

```python
from csv_to_hdf5 import csv_to_hdf5

csv_folder = "EODHD_Data"        # Set your CSV directory here
output_folder = "HDF5_output"    # Where to save HDF5 files
start_time = pd.Timestamp("2004-01-01 00:00:00", tz="UTC")
end_time = pd.Timestamp("2025-05-24 00:00:00", tz="UTC")  # Adjust as needed

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
csv_files = [
    f for f in glob.glob(os.path.join(csv_folder, "*.csv"))
]

for idx, f in enumerate(csv_files):
    stock_name = str(os.path.basename(f)).split("_")[0] #Our files are in format XXXX_intraday_1m.csv
    df = csv_to_hdf5(f, stock_name, idx, len(csv_files))

```

We now have 49.22 gb of HDF5 data in 503 files. And we soon found this would become one of the first problems when training, because we tried to train directly, without preprocessing the labels we wanted to use for each sequence of data, and with all these files adding a big overhead of I/O. And for the inference later it would also be a problem, having to read from all these files. Thus, we decided to merge the files into a single hdf5 file. 

Another decision was to remove closed-market hours. This would reduce a lot the noise because weekends, holidays and the closed market hours (from 16:30 to 09:00 US Eastern Time) would be removed. And to further remove noise, we decided to reduce from 503 stocks to the stocks that were already publicly traded in January 2004, reducing the number of stocks to 263. The final file with this data is 22.67 gb. 

NOTE: after issues with the training and needing to increase the number of layers of the transformer, the number of stocks was reduced to 100, with a file data of 8.5 gb. 

For this merge, we used the merge_hdf5_files.py. 

```python
from merge_hdf5_files import merge_hdf5_files

stock_names = ["AAPL","MSFT","TSLA"]
h5_paths = [f"HDF5_output/{stock_name}.h5" for stock_name in stock_names]
merged_h5_path = 'stocks_merged.h5'

merge_hdf5_file(stock_names,h5_paths,merged_h5_path)
```

Alright, now we have the stock data in one single file but… we still need to do some preprocessing because we want to train the model to learn “patterns” from sequences of data, and each sequence will need the labels that we expect our neural network to infer. We are entering the hyperparameter world already and we have not even exposed our model! But this is very important and actually model independent. Initially, we planned to go with the 503 stocks, with sequence length of 1200 (20 hours), with a stride of 10 minutes, with 6 features per step (open, high, close, low, volume and a NaN mask for those steps with data missing). But this turned to be too ambitious, even after all the preprocessing was done. Preprocessing the data of 20 years like this meant over 250 gb of data! And the training with 10 epochs of only one year took up 9 hours. So, valuing a bigger time span with less overlap of data between sequences, the data was structured in sequences of 390 steps (6.5 hours, which is the duration of the open market hours), with a stride of 185 (right in the middle of the open market hours). This resulted in a file of 26.32 gb, with 11345 sequences of 1 min interval stock data from 263 stocks.

NOTE: The file for 100 stocks is 9.67 gb. The number of sequences remain the same. 

For the sequences, the open high low close features were normalized by using the relative value with respect to the first step. So the first step would always be 0 for the 4 features and the rest would be the percentage of change with respect to that one. 

## The Labels

We have the features, but what about the labels? For the labels we needed to define two things: the labels per se, and the horizon, which is from how far from the closing value of the sequence we are calculating the value of the labels. For day trading purposes, we decided to use two horizons:

- 1 hour
- 1 day (in this case, one day is the same hour from the last sequence on the next open market)

The labels per horizon are 3, two classification labels and one regression label

Classification labels 

- Performance rank (Classification): How it ranks compared to all the stocks processed. Same for all horizons for simplicities sake. Values are:
    - 0: Top 80% performer
    - 1: 60% to 80% percentile
    - 2: 40% to 60% percentile
    - 3: 20% to 40% percentile
    - 4: Lowest 20% performers
- Risk (bool): States if a stock is risky by comparing the standard deviation of the return with a volatility threshold per horizon.
    - 1h: Risky if standard deviation above 2%
    - 1d: Risky if standard deviation above 3%

Regression labels 

- Return (float32): Relative predicted return with respect to close at last sequence step

This preprocessing can be done with the following code:

```python
from preprocess_merged_hdf5.py import process_all_stocks

merged_h5 = 'stocks_merged.h5'  # Path to merged HDF5 file
horizons = {
    '1h':   {"volatility_threshold": 0.02},
    '1d':   {"volatility_threshold": 0.03}
}
process_all_stocks(
            num_workers=12,
            seq_length=390,
            stride=185,
            horizons=horizons,
            merged_h5=merged_h5
        )
```

### The dataset

Thanks to the preprocessing done the dataset is easy to load, with the hdf5_sequence_dataset.py. Thanks to the preprocessing done, it is very quick to data reducing the I/O and memory at max. 

```python
from HDF_Sequence_Dataset import HDF5SequenceDataset

dataset = HDF5SequenceDataset('Data file.h5')
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
for x, y in loader:
     print(x.shape, y.shape)
dataset.close()

```

# The FORTUNE Transformer

![architecture horizontal.white background 3l.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/architecture_horizontal.white_background_3l.png)

Now that we have the data, we need a model that will learn from it. The arrival of transformer based neural networks has been disruptive specially in the Natural Language Model. So we wanted to test this technology in a different setting like financial technical analysis. Because we don’t need to extract series of data, an encoder only architecture has been selected to test it. It consist of:

- Linear embedding as input layer to project the features into a learned representation space
- A stock relative position embedding that learns the unique characteristics of each stock
- A positional encoder for the sequences to add temporal information
- The Temporal Transformer Encoder, which caputre temporal patterns in stock price sequences
- An attention pooling that aggregates all the temporal information from the sequences reducing the dimensionality
- A cross-stock multi head attention that learns relationships between different stocks
- The output layer, that predicts the labels with one head for each horizon

The FORTUNE Transformer can be found in the FORTUNE_Transformer.py, and our dataset contains 6 features (OHCL + Volume + NaN mask)

```python
from FORTUNE_Transformer import FORTUNETransformer

num_features = 6 #5 features + 1 NaN mask
 
  # Adjust num_features and num_stocks as needed based on your HDF5 data
model = FORTUNETransformer(num_features=num_features).to(device)
```

## **Training and evaluating the model**

To train the FORTUNE model we have to define our loss function carefully. In fact, our model is providing both classification and regression outputs, with diverse values. Thus, it is important that the loss is designed with this in mind. Our loss is based on the following function

$\mathcal{L}{\text{total}} = \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \left[ \sum_{h=1}^{H} \left( w_{\text{rank}} \cdot \mathcal{L}_{\text{CE}}^{(h)} + w_{\text{risk}} \cdot \mathcal{L}_{\text{BCE}}^{(h)} \right) + w_{\text{reg}} \cdot \sum_{h=1}^{H} \mathcal{L}_{\text{Huber+Sign}}^{(h)} \right]$ 

It is composed of the weighted combination of the losses for each label. Each label has its own type of loss due to their nature:

- Ranking label: Cross-Entropy Loss, which is a multiclass classification loss

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum{i=1}^{N} \sum_{c=1}^{5} y_{i,c} \log p_{i,c}
$$

- Risk label: Binary Cross-Entropy Loss,, useful for binary classification problems

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum{i=1}^{N} \left[ y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i)) \right]
$$

- Huber-loss with sign penalty: which combines the huber loss, which is a good regression loss robust to outliers and is quadratic for small errors, with a sign penalty that penalizes the predicted return if it has the wrong sign

$$
\mathcal{L}_{\text{Huber+Sign}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \text{Huber}(\hat{y}_i, y_i) \cdot w_{\text{huber}} + \left(1 - \mathbb{I}\left[\text{sign}(\hat{y}_i) = \text{sign}(y_i)\right]\right) \cdot w_{\text{sign}} \right]
$$

where

$$
\text{Huber}(\hat{y}_i, y_i) = \begin{cases} 0.5, (\hat{y}_i - y_i)^2 & \text{if } |\hat{y}_i - y_i| \leq \delta \\ \delta, |\hat{y}_i - y_i| - 0.5, \delta^2 & \text{otherwise} \end{cases} 
$$

The most important hyperparameters from this Loss are the weights, which need to be tuned in order that the losses are influencing the training in a balanced way. 

Other metrics we will use to monitor the training are the following:

### Standard accuracy for the ranking label only (not good for binary classification)

$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

### Balanced accuracy for the risk label only (only for binary classification)

$$
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)
$$

### Macro-F1 score for the ranking classification tasks and F1 Score binary for risk classification

$$
F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C}F1_c=\frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c }

$$

$$
F1 =\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} }

$$

Where

$$

\text{Precision} = \frac{TP}{TP + FP }\quad \text{Recall} = \frac{TP}{TP + FN }

$$

### SMAPE (Symmetric Mean Absolute Percentage Error) for the future returns label

Selected over MAPE because it handles very small values more gracefully, it gives an interpretable percentage-based measure of how close the predicted values are to the actual values. 

$$

\text{SMAPE} = \frac{100}{N} \sum_{i=1}^{N} \frac{\left| \hat{y}_i - y_i \right|}{\left( \left| y_i \right| + \left| \hat{y}_i \right| \right)/2 + \varepsilon}

$$

These table summarizes the metrics and what values we can expect to consider we have a good model

| **Metric** | **What It Measures** | **Tells You** | **Good Value (Typical Range)** |
| --- | --- | --- | --- |
| **Standard Accuracy** | % of correct predictions among all predictions (for classification) | How often the model predicts the **exact class correctly** | 40–60% for 5-class |
| **Balanced Accuracy** | Mean of sensitivity (recall for class 1) and specificity (recall for class 0) | Whether the model is performing **fairly across imbalanced classes**, especially for binary risk labels | > 70% for balanced binary; ≥ 50% is decent if imbalance is high |
| **Macro F1 Score** | Average F1 score across all classes, treating each class equally | How well the model performs **on all classes**, including rare ones (important for ranking) | > 0.50 is acceptable;
> 0.65 is strong |
| **F1 Score** | Harmonic mean of precision and recall (for binary classification) | How well the model balances **false positives and false negatives** on the **risk** task | > 0.70 is good; 
> 0.80 is very strong |
| **SMAPE** | Symmetric mean absolute percentage error (for regression) | The **average % error** in predicted return magnitudes, normalized and symmetric | < 25% is decent; 
< 15% is excellent |

## Training and Hyperparameters

The hyperparameters can be found on the [parameters.py](http://parameters.py) file

```python
horizons_390 = {
    '1h':   {"volatility_threshold": 0.02}, 
    '1d':   {"volatility_threshold": 0.03}
}

h_params = {
    "horizons": horizons_390,    
    "learning_rate": 1e-4,
    "epochs": 50,
    "batch_size": 25,
    "chunk_size": 50,   # This puts the nu
    "d_model": 128,     # Defines the embedding dimension
    "num_layers": 3,    # Defines the number of transformer layers
    "nhead": 8,         # Defines the attention heads for the stock attention
    "regression_loss_weight": 1,  # Sets the weight of the regression in the lost
    "classification_loss_weight": [1, 1], #Weight for ranking and risk label loss
    "sign_penalty_weight": 3,   #Defines the weight of the sign in the huber loss
    "huber_weight": 1,   # Defines the weigh of huber with respect to the sign
    "huber_delta": 0.02,  # Defines the huber delta
    "weight_decay": 5e-3 # Utilizado en el optimizador AdamW
}
```

As optimizer, the AdamW was selected to try to help with the weight calculations. It combines the Adam optimizer (which adapts learning rates for each parameter) with a decoupled weight decay (L2 regularization). There is also a scheduler, to adjust the learning rate of the optimizer.

Because of the amount of parameters to be calculated (1,860,751!), some memory strategies needed to be applied, specially when training the transformer part. 

## The hardware

For training, we have used a Google Cloud VM with an Intel Cascade Lake g2-standard with 16 vCPUs, 64 GB of RAM, and an NVIDIA GPU l4 with 24 GB of VRAM. 

## **The parameters**

It is possible to see how many parameters are trainable by executing the file num_parameters.py. For our current model:

Feature embedding: 896
Stock embedding: 12,800
Transformer: 1,779,072
Pooling: 129
Cross attention: 66,048
Output heads: 1,806
**Total: 1,860,751 parameters**

## Running the training

Training can be found in the file training.py. 

```python
from training import training

training(train_split = 0.4,dataset_file = 'dataset_file.h5')
```

During the training we create checkpoints of the model with the _checkpoint.pt suffix and the final model with the [final.pt](http://final.pt) suffix. The training will detect if a final.pt file already exists and prompt if the training wants to be restarted or just evaluate the model. It also detects if a checkpoint file exists and will prompt it you want to resume the training with that model or start a whole new training. In every epoch of the training, it will plot the metrics in the folder ./plot/{timestamp}/epoch_{num} 

# The Results (spoiler: we won’t be rich)

The final results presented are based on the following hyperparameters:

```python
h_params = {
    "horizons": horizons_900,
    "learning_rate": 1e-4,
    "epochs": 50,
    "batch_size": 25,
    "d_model": 128,
    "num_layers": 3,
    "nhead": 8,
    "chunk_size": 50,
    "huber_delta": 0.02,
    "regression_loss_weight": 3,
    "classification_loss_weight": [1, 0.5],
    "sign_penalty_weight": 4,
    "huber_weight": 1,
    "weight_decay": 5e-3
}
```

The results show that the current set up… do not work. It completely underfits and does not get to learn. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/33d2861e-6ba4-470f-bd08-13025e561e8f.png)

Many tests have been done with different type of hyperparameters and labels, with a focus on the loss weights, but in general, the model underfits. 

The training and validation loss might seem to be slightly decreasing but it does not go much further than that. Each epoch takes around 15 minutes to be trained, and several 20 (5 hours) and 50 epoch (12.5 hours) trainings have been done. But the results are similar. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/e47ac154-a187-41c6-a9b2-d038bcc00fd4.png)

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%203.png)

In this last iteration, we wanted the regression loss to have a higher impact than the risk and ranking losses, trying to make the model specialize in it, as it included the sign penalty, which helps with the buy/sell signal we really want to get. That’s why the weight for the regression is higher. But the system does not really get to learn. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%204.png)

For the ranking label, it seems it perferms a bit better than random, which would be 20% accuracy, but very far from the 40%-60% that we would at least expect to be ok.

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%205.png)

For the returns, the SMAPE shows also how far we are from the actual returns. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%206.png)

The return sign accuracy also shows a zero learning curve. , stuck around 0.4% of accuracy. This is ok, since the data is not balanced (around 40% negative returns and 60% positive returns), so it makes sense that it is not in the 50%. 

A new training with different hyperparameters, with higher importance on the regression and almost ignoring the risk, showed a bit of promise in the regression loss for 1h horizon, which makes sense that it works better than the 1d. This indicates a focus on the 1-hour horizon, and regression only could be interesting. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%207.png)

## Trading Bot

The predictions from the model above are integrated into an automated trading strategy designed to trade stocks autonomously. Currently, the strategy is limited to buying and selling actions. The trading algorithm operates as follows:

### Operational Workflow

The trading bot operates by processing market data through a transformer model/trading data to generate predictions, which inform trading decisions. Below are the steps involved:

1. **Data Collection and Preprocessing**
    
    We use historical data as explained earlier (e.g., price, volume, technical indicators). Preprocess data by normalising prices, handling missing values, and creating sequences for the transformer model input.
    
2. **Transformer Model Prediction**
    
    Feed preprocessed data into a trained transformer model. The model predicts future price movements over a specified horizon (e.g., next hour or day). In case of model unavailability, we have also used the shifted real trading data itself for this purpose.
    
3. **Signal Generation**
    
    Convert model predictions into trading signals. For example, a predicted price increase above a threshold (e.g., 1%) triggers a "buy" signal, while a predicted decrease triggers a "sell" signal. Include a confidence threshold to filter low-probability predictions.
    
4. **Risk Management**
    
    Apply risk management rules, such as position sizing based on account balance (e.g., risking 1% per trade). Ensure compliance with portfolio constraints (e.g., maximum exposure).
    
5. **Trade Execution**
    
    Execute trades via an API based on signals (future work). Log trade details (entry price, volume, timestamp) for analysis.
    
6. **Performance Monitoring and Feedback**
    
    Continuously monitor open positions and market conditions. Update the model with new data periodically to adapt to changing market dynamics. Log performance metrics (e.g., profit/loss, win rate).
    

### Code Snippets

The tradingbot feature can be accessed from tradingbot.py. The AdaptiveHorizonTrader takes in initial capital, transaction cost, slippage, maximum_positions, etc, as initial parameters. Below are key code snippets from the AdaptiveHorizonTrader class, illustrating each workflow step. 

### 1. Initialisation and Data Collection

```python
# Initialize trader with parameters
trader = AdaptiveHorizonTrader(
    initial_capital=100000,
    transaction_cost=0.0008,  # 0.08% per trade
    slippage=0.0003,         # 0.03% slippage
    max_positions=4,
    min_allocation=0.03,     # 3% minimum
    max_allocation=0.35      # 35% maximum
)

# Collect mock price and prediction data
tickers = [str(i) for i in range(263)]
dates = pd.date_range(start='2023-01-01', end='2023-02-11', freq='1h')
price_data = {t: pd.DataFrame({'close': prediction_temp[:,int(t),-1]}, index=dates) for t in tickers}
prediction_data = pd.DataFrame({t: [prediction_temp[i,int(t),:] for i in range(len(dates))] for t in tickers}, index=dates)
```

**2. Data Preprocessing**

```python
# Structure prediction data for trading (1h_return, 1h_risk, 1d_return, 1d_risk, current_price)
prediction_data = pd.DataFrame({
    t: [prediction_temp[i,int(t),:] for i in range(len(dates))]
    for t in tickers
}, index=dates)
```

**3. Prediction-Based Asset Ranking**

```python
def rank_assets(self, predictions: np.ndarray, horizon: str) -> List[Tuple]:
    horizon_idx = 2 if horizon == '1h' else 2
    risk_idx = 1 if horizon == '1h' else 3
    ranked = []
    for i in range(predictions.shape[1]):
        ret = predictions[0,i][horizon_idx]
        risk = predictions[0,i][risk_idx]
        price = predictions[0,i][6]
        score = ret * (1 - 0.3*risk)  # Risk-adjusted score
        ranked.append((i, ret, risk, price, score))
    return sorted(ranked, key=lambda x: x[4], reverse=True)
```

**4. Position Sizing and Risk Management**

```python
def calculate_position_weights(self, assets: List[Tuple]) -> np.ndarray:
    returns = np.array([x[1] for x in assets])
    risks = np.array([x[2] for x in assets])
    ret_range = returns.max() - returns.min()
    norm_returns = (returns - returns.min()) / (ret_range + 1e-8)
    risk_mult = 1.0 - (0.3 * risks)
    scores = norm_returns * risk_mult
    temp = 0.5
    exp_scores = np.exp(scores / temp)
    weights = exp_scores / exp_scores.sum()
    weights = np.clip(weights, self.min_allocation, self.max_allocation)
    return weights / weights.sum()
```

**5. Trade Execution**

```python
def execute_trade(self, predictions: np.ndarray, tickers: List[str], timestamp: datetime, horizon: str) -> Dict:
    ....
    return trade_record
```

**6. Performance Monitoring and Feedback**

```python
def liquidate_positions(self, actual_prices: np.ndarray, tickers: List[str], timestamp: datetime) -> Dict:
    .....
    return trade_record

def get_performance_report(self) -> Dict:
    ....
    return {
        'initial_capital': self.initial_capital,
        'final_capital': self.current_capital,
        'total_return_pct': (self.current_capital - self.initial_capital)/self.initial_capital*100,
        'total_trades': len(closed_trades),        'win_rate': win_rate,
        'avg_return': returns.mean(),
        'median_return': returns.median(),
        'max_drawdown': (returns.cumsum().cummax() - returns.cumsum()).max(),
        'sharpe_ratio': returns.mean()/(returns.std() + 1e-8)*np.sqrt(252),
        'profit_factor': sum(t['total_pnl'] for t in winning_trades)/abs(sum(t['total_pnl'] for t in closed_trades if t['total_pnl'] < 0)),
        'total_transaction_cost': sum(t.get('transaction_cost',0) for t in self.trade_history)
    }
```

These snippets demonstrate how the AdaptiveHorizonTrader class implements a trading bot that uses transformer model predictions to rank assets, allocate capital, execute trades, and monitor performance, ensuring a robust and adaptive trading strategy.

### Backtesting

Backtesting evaluates the trading bot’s performance using historical data over a 1-year period. The process is:

1. **Setup**
    
    Start with a fixed initial capital (e.g., €100,000). Simulate trades twice daily (current sequence length is for half a day), using historical data to generate transformer model predictions. Rank stocks by expected performance, select the top 10, and reinvest all capital after selling current holdings at each interval. Include transaction costs (e.g., 0.1% per trade). Unfortunately, the prediction model has not yet reached a mature stage, so to demonstrate the effectiveness of the trading bot, the original future trading data itself is used. Using this data will result in enormous profit, which won't say anything about the efficacy of our trading bot. Hence, the trading data is delayed by a few hours, such that if we are trading for prediction after 1 hour, we will use trading data after 10 hours to calculate profit and loss. This is a temporary workaround to show the features of the trading bot.
    
2. **Trading Logic**
    - Buy the top 4 ranked stocks at the decision time based on predictions.
    - Hold positions until the prediction horizon ends (e.g., 1 hour or 1 day).
    - Sell all positions at the horizon’s end and calculate P&L using actual prices.
    - Update capital and repeat for the next trading interval.
3. **Performance Evaluation**
    
    Plotting the **Equity Curve**, **Drawdown Plot**, and **Trade Distribution Plot** is essential for evaluating a trading bot’s performance, as each provides unique insights:
    
    - **Equity Curve**: Shows portfolio value over time, revealing overall profitability and growth trends. It helps assess whether the strategy generates consistent gains or experiences volatility.
    - **Drawdown Plot**: Displays percentage declines from peak portfolio value, highlighting risk exposure and worst-case losses. It’s critical for understanding the strategy’s risk profile and capital preservation.
    - **Trade Distribution Plot**: Illustrates the frequency and spread of trade returns, indicating consistency and identifying outliers (e.g., large wins/losses). It helps evaluate the reliability of the strategy’s performance.
    
    Together, these plots provide a comprehensive view of profitability, risk, and consistency, enabling traders to optimise the AdaptiveHorizonTrader strategy.
    
    ![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%208.png)
    

The equity curve shows that the portfolio made profits at certain points, after which it started to make losses.

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%209.png)

The drawdown plot shows a decrease in portfolio values by 50% from its peak.

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%2010.png)

The trade distribution histogram shows that most of the transactions resulted in negligible returns. This histogram would be more meaningful when the actual prediction from the model is used.

The overall analysis of the trading bot is shown below:
initial_capital: 100000
final_capital: 63615.3203125
total_return_pct: -36.384681701660156
total_trades: 776
win_rate: 0.5000
avg_return: 70.42601013183594
median_return: 2.4281492233276367
max_drawdown: 39780.30859375
sharpe_ratio: 1.3012
profit_factor: 1.2590093612670898
total_transaction_cost: 91225.7734375

### Performance of the trading bot when exact trading data is used

All three plots below show values as expected, which are insane profit margins. The trade distribution histogram remains positive. If nothing else, this demonstrates that the trading bot works correctly in principle. 

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%2011.png)

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%2012.png)

![image.png](FORTUNE%20Financial%20Outlook%20Realized%20by%20Unified%20Netw%2022cfd22a31fe806faeb9cbbe1d6d7036/image%2013.png)

# **Conclusions and next steps**

There are several conclusions that can be extracted from the results:

- The selected labels are not useful or are not well weighted in the loss
- Too complex to have several horizons at once
- Training transformers with large amounts of data requires very high computing capacities, creating a bottleneck
- The cross-stock attention might give an overhead that does not provide as much benefit
- Maybe, technical analysis is really not a good active investing strategy

Next steps:

- Simplify
- Simplify
- Simplify a bit more
- Then simplify
- Rethink the labels, with no multiple types of loss combined.
- Try to train with sequences of data, without the stock comparison, to see if it helps with the technical analysis
- No horizons first, stick to only one, reducing complexity.
- Test against other models such as LSTMs
