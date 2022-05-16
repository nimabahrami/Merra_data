import argparse
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to csv file",
                        type=str)
    parser.add_argument("--metric", help="metric e.g. Wind speed",
                        type=str)
    parser.add_argument("--height", help="height for speed",
                        type=float)
    parser.add_argument("--plotres", help="plot for results Y or y for plotting",
                        type=str)
    parser.add_argument("--seasontype", help="type of seasonality -additive or multiplicative",
                        type=str)
    parser.add_argument("--roundval", help="round values to specific decimal point",
                        type=int)
    parser.add_argument("--rmnan", help="check if should remove nan type Y or y for removing",
                        type=str)
    
    args = parser.parse_args()

    data = pd.read_csv(args.path, sep=',')
    labels = list(data.iloc[21, 0].split(';'))
    print(labels)
    metric = args.metric if args.metric else 'Wind speed'
    index_of_speed = [(inx, item)
                      for inx, item in enumerate(labels) if item == metric][0]
    new = pd.DataFrame(columns=labels)
    data = data.iloc[22:, 0].to_frame()
    result = pd.concat([data, new])
    data[labels] = data.iloc[:, 0].str.split(';', expand=True)
    windspeed = data.iloc[:, index_of_speed[0]+1]
    
    if args.rmnan in list('Yy'):
        windspeed = windspeed.fillna(0)
        
    windspeed = pd.to_numeric(windspeed)

    if args.height:
        windspeed = windspeed*(args.height/10)**0.21
    
    if args.seasontype:
        res = seasonal_decompose(windspeed, model=args.seasontype, period=720)
    else:
        res = seasonal_decompose(windspeed, model='additive', period=720)

    removed_seasonality = windspeed-res.seasonal
    if args.roundval:
        removed_seasonality = removed_seasonality.round(args.roundval)
    removed_seasonality.to_csv('result.csv', index=False, header=False)
    print('saved as result.csv')

    if args.plotres in list('Yy'):
        pyplot.plot(removed_seasonality)
        res.plot()
        pyplot.show()
