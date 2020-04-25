# covid19

## Intro/Lit Review (current state of covid19 prediction, and how we are going to contribute)

## Data: The unique data set we were able to compile

### [Weather](https://developer.weathersource.com/tools/postman-collection-onpoint-api/)

There is a large number of columns here. It would be good to regularize the data set and identify the most important columns before writing up the variable description. The full set of variable descriptions are here. 

[Variable description](https://developer.weathersource.com/documentation/resources/get-points-onpoint_id-history/)

### [Potential Option for Crowd Movement](https://github.com/COVIDExposureIndices/COVIDExposureIndices?utm_source=wechat_session&utm_medium=social&utm_oi=667254872605331456#exposure-indices-derived-from-placeiq-movement-data)

### [Google Mobility Report Extraction](https://github.com/kylemcdonald/covid-mobility-data)
Currently Google provides the raw data, but the data is weeks behind the NYT COVID data. The latest at the moment is 4/10/2020.

### [Zip Code](https://simplemaps.com/data/us-zips)

### [Census Income Data](https://www.census.gov/data/tables/2019/demo/income-poverty/p60-266.html)
Income data by county has been merged to the time series data. Check the column "income_2018" in `time_series_all.csv`.

### [Population]
Population for 2018 (infered) data has been merged to the time series data. Check the "pop_2018" in `time_series_all.csv`.

### [Yelp](https://www.yelp.com/fusion)



#### Descriptions

#### Aggregation Methodology

#### Notes/Caveats

## Inference: Identify factors affecting the spread of disease

## Prediction: Predict the spread on the county level


number of cases on day n
NTH: start date of cases

## Benchmark: Compare predictive methods in disease prediction

