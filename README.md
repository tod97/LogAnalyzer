# LogAnalyzer

I've created this repo to try the features proposed from the [logpai team](http://www.logpai.com).
What they are offering is some repos that are meant to analyze different server's logs.
This repo groups all their repos in one:

- [Loghub](https://github.com/logpai/loghub): where you can find all the available datasets;
- [Logparser](https://github.com/logpai/logparser): where you can find all the available datasets parsers to convert logs into structured datas;
- [Loglizer](https://github.com/logpai/loglizer): where you can find all the different models to analyze the log_structured;

In the project root you can find some working examples and a benchmark for all the analyzers.

## Steps

> I focused on the HDFS_1 dataset and MLP as analyzer

1. Download the HDFS_1 dataset from [Loghub repo](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_1)
2. Split the log file (~1.6 GB) into blocks of the size you prefer. Unix solution:

```sh
split -b 200m HDFS.log
```

3. Edit the `log_file` variable inside "parser_hdfs.py":

```py
log_file   = 'HDFS_100m.log'  # The input log file name
```

4. Run the parser_hdfs to create a new log_structure file:

```sh
python parser_hdfs.py
```

5. Edit the `struct_log` variable inside benchmark's notebook to match your new log_structured file name:

```py
struct_log = './loghub/HDFS/HDFS_100m.log_structured.csv' # The benchmark dataset
```

6. Have fun :)
