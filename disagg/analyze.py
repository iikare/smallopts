#!/usr/bin/env python3
from datetime import datetime
import pandas as pd
import json
import glob
import csv
import ast
import re
import numpy as np
import sys
import os

arg1 = sys.argv[1]
arg2 = sys.argv[2]

def extract_tokens(request_info_str):
    obj = json.loads(request_info_str)
    inner = obj.get("data", "")

    if isinstance(inner, str) and inner.startswith('['):
        try:
            parsed_list = ast.literal_eval(inner)
            if parsed_list and isinstance(parsed_list[0], str):
                inner = parsed_list[0]
        except Exception:
            pass  # fall back to regex on the original string

    kv = dict(re.findall(r'(\w+)=([0-9]+)', inner))

    prompt_tokens = int(kv["prompt_tokens"])
    output_tokens = int(kv["output_tokens"])
    return prompt_tokens, output_tokens

def clean_part(x):
    if not isinstance(x, str):
        return ""
    x = x.strip()
    if not x or x.startswith("Unnamed:"):
        return ""          # treat these as empty
    return x

def combine_levels(col_tuple):
    # col_tuple looks like ('Run Info', 'Run ID', 'Unnamed: 2_level_2')
    parts = [clean_part(x) for x in col_tuple]
    parts = [p for p in parts if p]   # drop empties
    return " ".join(parts)            # e.g. "Run Info Run ID"

def extract_csv_columns(csv_file, columns, output_file=None):
    df = pd.read_csv(csv_file, header=[0,1,2])

    # Apply to all columns
    df.columns = [combine_levels(col) for col in df.columns]

    # (Optional) drop any columns that ended up with an empty name
    #df = df.loc[:, df.columns != ""]

    print("1.")
    print(df.columns)

    print("2.")
    print(df.shape)
    
    if output_file:
        df.to_csv(output_file, index=False)
    # df.to_csv("tmp.csv", index=False)
    
    return df

if __name__ == "__main__":
    old_cols = ['Request Loader', 'Duration', 'Name', 
            'Total Requests', 'Total Requests per second mean', 'Total Requests per second median', 'Total Requests per second std dev', 'Total Requests per second [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]',
            'Total Request latency mean', 'Total Request latency median', 'Total Request latency std dev', 'Total Request latency [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]', 
            'Total Time to first token ms mean', 'Total Time to first token ms median', 'Total Time to first token ms std dev', 'Total Time to first token ms [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]', 
            'Total Inter token latency ms mean', 'Total Inter token latency ms median', 'Total Inter token latency ms std dev', 'Total Inter token latency ms [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]',
            'Total Output tokens per second mean', 'Total Output tokens per second median', 'Total Output tokens per second std dev', 'Total Output tokens per second [min, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, max]']


    model_path = os.path.join("out", arg1, "model")
    with open(model_path, "r") as model_file:
        model = model_file.read().strip()

    cols = ['Run Info Run ID', 'Timings Duration Sec', 'Run Info Profile',
            'Request Counts Total', 
            'Server Throughput Successful Requests/Sec Mean', 'Server Throughput Successful Requests/Sec Median', 
            'Server Throughput Successful Requests/Sec Std Dev', 'Server Throughput Successful Requests/Sec Percentiles',
            'Request Latency Successful Sec Mean', 'Request Latency Successful Sec Median', 
            'Request Latency Successful Sec Std Dev', 'Request Latency Successful Sec Percentiles',
            'Time to First Token Successful ms Mean', 'Time to First Token Successful ms Median', 'Time to First Token Successful ms Std Dev', 
            'Time to First Token Successful ms Percentiles',
            'Inter Token Latency Successful ms Mean', 'Inter Token Latency Successful ms Median', 'Inter Token Latency Successful ms Std Dev',
            'Inter Token Latency Successful ms Percentiles',
            'Token Throughput Successful Total Tokens/Sec Mean', 'Token Throughput Successful Total Tokens/Sec Median',
            'Token Throughput Successful Total Tokens/Sec Std Dev', 'Token Throughput Successful Total Tokens/Sec Percentiles',
            'Timings Duration Sec',
            'Run Info Requests',
            'Run Info Run Index',
            'Token Metrics Successful Input Tokens Mean',
            'Token Metrics Successful Output Tokens Mean',
            'Token Throughput Successful Input Tokens/Sec Mean', 'Token Throughput Successful Input Tokens/Sec Median',
            'Token Throughput Successful Input Tokens/Sec Std Dev', 'Token Throughput Successful Input Tokens/Sec Percentiles',
            'Token Throughput Successful Output Tokens/Sec Mean', 'Token Throughput Successful Output Tokens/Sec Median',
            'Token Throughput Successful Output Tokens/Sec Std Dev', 'Token Throughput Successful Output Tokens/Sec Percentiles',
            ]

    cols_new = ['model', 'duration', 'name', 
                'total_requests',
                'total_requests_mean',
                'total_requests_median',
                'total_requests_stdev',
                'total_requests_qrt',
                'e2e_mean',
                'e2e_median',
                'e2e_stdev',
                'e2e_qrt',
                'ttft_mean',
                'ttft_median',
                'ttft_stdev',
                'ttft_qrt',
                'itt_mean',
                'itt_median',
                'itt_stdev',
                'itt_qrt',
                'tps_mean',
                'tps_median',
                'tps_stdev',
                'tps_qrt',
                'duration',
                'input_tokens_expected',
                'output_tokens_expected',
                'input_tokens_actual_mean',
                'output_tokens_actual_mean',
                'input_tps_mean',
                'input_tps_median',
                'input_tps_stdev',
                'input_tps_qrt',
                'output_tps_mean',
                'output_tps_median',
                'output_tps_stdev',
                'output_tps_qrt',
                ]

    print(f'expected: {len(cols)}, actual: {len(cols_new)}')

    cols_final = ['id', 'model', 'name', 'duration', 'total_requests']
    for label in (['total_requests', 'e2e', 'ttft', 'itt', 'tps', 'input_tps', 'output_tps']):
        for q in (['mean','stdev','min','p1','p25','p50','p75','p99','max']):
            cols_final.append(f'{label}_{q}')



    import glob, os
    all_data = []
    csv_path = os.path.join("out", arg1, "*.csv")
    for file in glob.glob(csv_path):
        file_data = []
        print(file)
        df = extract_csv_columns(file, cols)
        df = df.rename(columns=dict(zip(cols, cols_new)))
        for index, row in df.iterrows():
            file_data.append(datetime.now().strftime('%-m/%-d/%Y %-H:%M'))
            file_data.append(model)
            conc_json = json.loads(row['name'])
            file_data.append(conc_json["completed_strategies"][0]["max_concurrency"])
            file_data.append(row['duration'])
            file_data.append(row['total_requests'])

            for label in (['total_requests', 'e2e', 'ttft', 'itt', 'tps']):
                file_data.append(row[f'{label}_mean'])
                file_data.append(row[f'{label}_stdev'])
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[0]) # min
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[2]) # p1
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[5]) # p25
                file_data.append(row[f'{label}_median'])        # p50
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[6]) # p75
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[8]) # p99
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[9]) # max
            file_data.append(row['duration'])
            p, o = extract_tokens(row["input_tokens_expected"])
            file_data.append(p)
            file_data.append(o)
            file_data.append(row['input_tokens_actual_mean'])
            file_data.append(row['output_tokens_actual_mean'])
            for label in (['input_tps', 'output_tps']):
                file_data.append(row[f'{label}_mean'])
                file_data.append(row[f'{label}_stdev'])
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[0]) # min
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[2]) # p1
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[5]) # p25
                file_data.append(row[f'{label}_median'])        # p50
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[6]) # p75
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[8]) # p99
                file_data.append(ast.literal_eval(row[f'{label}_qrt'])[9]) # max

        all_data.append(file_data)
        print(file_data)
        print(f'expected: {len(cols_final)}, actual: {len(file_data)}')
        # break

    out_file = arg2
    with open(out_file,"w+") as csv_file:
        csv_w = csv.writer(csv_file, delimiter=',')
        #csv_w.writerow(cols_final)

        csv_w.writerows(all_data)

    print(f"postprocessing complete! output saved at {out_file}")
