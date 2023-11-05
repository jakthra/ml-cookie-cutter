# ml-cookie-cutter
Base tooling for ML projects from data ingress to model performance monitoring. Using an ELT design pattern for data ingress to version, snapshot, observe and otherwise transform "raw" input data.

![overview](docs/assets/ml-cookie-cutter.png)

## Behavioural applications

### Data

Load data into local datalake using annotated structures
```
ml load-dataset -n timeseries-example
```

List datasets and their status
```
ml datasets
```

Transform data and generate consumable datasets using

```
dbt build
```

### ML

train models using

```
ml experiment dataset=amazing_dataset_v1 model=ts_trs_small
```

### Ops

```
ml build --run-name incredible-crane-313
```



## Folder structure

📦data
 ┣ 📂transformed
 ┗ 📂raw
📦ml_cookie_cutter
┣ 📂data
┣ 📂ml
┗ 📂ops
📦tests
📦docs
 ┗ 📂assets

