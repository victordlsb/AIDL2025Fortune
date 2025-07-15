# FORTUNE

Financial Outlook Realized by Unified Network Encoder

- Formatted text in the Github README of your repository.
- Add link to repo from presentation front slide.
- Must contain instructions to run the code
- Must contain per each experiment
    - Hypothesis: what do you expect to learn/observe from the experiment?
    - Experiment setup: here you explain what the experiment consists on (architecture, data...)
    - Results: pretty obvious
    - Conclusions: what insights do you get from the results (usually these lead to new hypothesis)

What NOT to do:

- The code alone will not count as Final Report, even documented code
- The slides must be submitted as official deliverable but will not be considered Final Report

# Introduction

Beating the stock market is a wish that any investor has dreamed of and very few have been able. There are several ways of trading, but when it comes to stock trading, strategies can be divided between active and passive. 

**Active strategies** involve frequent buying and selling of securities with the goal of outperforming the market or a specific benchmark. These strategies rely on market analysis, timing, and the skill of the investor or fund manager to identify short-term opportunities. Day trading based on technical analysis is a classic example of an active strategy. 

![Some examples of stock patterns used in technical analysis](FORTUNE%2022cfd22a31fe806faeb9cbbe1d6d7036/81d49b50-c112-4065-ace6-0948237a93ca.png)

Some examples of stock patterns used in technical analysis

In contrast, passive strategies aim to match the performance of a market index rather than beat it. Investors typically buy and hold a diversified portfolio over the long term, such as through index funds or Exchange Traded Funds (ETFs). Passive investing involves minimal trading and usually comes with lower fees, making it a popular choice for long-term wealth building. 

![S&P 500 returns in 30 years](FORTUNE%2022cfd22a31fe806faeb9cbbe1d6d7036/image.png)

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

![image.png](FORTUNE%2022cfd22a31fe806faeb9cbbe1d6d7036/image%201.png)

These are usually displayed using candlestick charts, where each candlestick represents the time step. The timestep can be from 1 minute to days, months or even years. The resolution really depends on what you are analysing. For day trading, 1 minute or 5 minute timesteps are used. 

![image.png](FORTUNE%2022cfd22a31fe806faeb9cbbe1d6d7036/image%202.png)

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

For this merge, we used the merge_hdf5_files.py. 

```python
from merge_hdf5_files import merge_hdf5_files

stock_names = ["AAPL","MSFT","TSLA"]
h5_paths = [f"HDF5_output/{stock_name}.h5" for stock_name in stock_names]
merged_h5_path = 'stocks_merged.h5'

merge_hdf5_file(stock_names,h5_paths,merged_h5_path)
```

Alright, now we have the stock data in one single file but… we still need to do some preprocessing because we want to train the model to learn “patterns” from sequences of data, and each sequence will need the labels that we expect our neural network to infer. We are entering the hyperparameter world already and we have not even exposed our model! But this is very important and actually model independent. Initially, we planned to go with the 503 stocks, with sequence length of 1200 (20 hours), with a stride of 10 minutes, with 6 features per step (open, high, close, low, volume and a NaN mask for those steps with data missing). But this turned to be too ambitious, even after all the preprocessing was done. Preprocessing the data of 20 years like this meant over 250 gb of data! And the training with 10 epochs of only one year took up 9 hours. So, valuing a bigger time span with less overlap of data between sequences, the data was structured in sequences of 390 steps (6.5 hours, which is the duration of the open market hours), with a stride of 185 (right in the middle of the open market hours). This resulted in a file of 26.32 gb, with 11345 sequences of 1 min interval stock data from 263 stocks.

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

# The FORTUNE Transformer

![architecture horizontal.white background.png](FORTUNE%2022cfd22a31fe806faeb9cbbe1d6d7036/architecture_horizontal.white_background.png)

Now that we have the data, we need a model that will learn from it. The arrival of transformer based neural networks has been disruptive specially in the Natural Language Model. So we wanted to test this technology in a different setting like financial technical analysis. Because we don’t need to extract series of data, an encoder only architecture has been selected to test it. It consist of:

- Linear embedding as input layer to project the features into a learned representation space
- A stock relative position embedding that learns the unique characteristics of each stock
- A positional encoder for the sequences to add temporal information
- The Temporal Transformer Encoder, which caputre temporal patterns in stock price sequences
- An attention pooling that aggregates all the temporal information from the sequences reducing the dimensionality
- A cross-stock multi head attention that learns relationships between different stocks
- The output layer, that predicts the labels with one head for each horizon

The FORTUNE Transformer can be found in the FORTUNE_Transformer.py

**Training and evaluating the model**

To train the FORTUNE model we have to define our loss function carefully. In fact, our model is providing both classification and regression outputs, with diverse values. Thus, it is important that the loss is designed with this in mind. 

Accuracy calculation

Only for the binary classification

Using balanced accuracy:

$$
balanced accuracy = (sensitivity + specificity)/2
$$

where 

$$
sensitivity = true positives / (true positivies + false negatives)
$$

$$
specificity = true negatives / (true negatives + false positives)
$$

It is calculated per label and then averaged for globally

F1 score for the binary classification tasks:

Calculating the F1 score globally (for all labels at once) and per label, where F1 score is

$$
F1Score = 2 * (precision*sensitivity) /(precision + sensitivity)
$$

Where

$$
precision = truepositives/(truepositives+falsepositives)
$$

For regression model metrics:

Symmetric Mean Absolute Percentage Error

$$
SMAPE = 100 * mean(|targets - predictions|/((targets + predictions)/2))
$$

Hyperparameters

Started with 500 stocks, stride 10, then moved to 60. Seq length 1200 moved to 900. 

Finally, done 390 seq len, stride 185 for half day overlaps 

From 5 labels (up, top, worst, risk, return) to 3 (top/worst, risk, return)

### Trading Bot

The predictions from the model above are integrated into an automated trading strategy designed to trade stocks autonomously. Currently, the strategy is limited to buying and selling actions. The trading algorithm operates as follows:

- We begin with a fixed initial capital, e.g., €1000.
- Trades are executed twice daily (this frequency can be adjusted, e.g., to hourly).
- Based on model predictions, stocks are ranked by their expected performance.
- The portfolio is limited to the top 10 ranked stocks.
- At each trading interval, all current holdings are sold, and the capital is reinvested into the new top 10 stocks.
- To evaluate long-term performance, the strategy is backtested over a 2-year period and compared against investing the same initial amount in the S&P 500.
- 
1. Buy stocks at decision time based on predictions
2. Hold exactly until the predicted horizon (1h or 1d)
3. Sell all positions at horizon end
4. Calculate P&L based on actual prices at horizon time
5. Start fresh with the updated capital for the next trade
