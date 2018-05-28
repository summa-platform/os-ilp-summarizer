#!/bin/env python3
import os
from glob import glob
import concurrent.futures
import urllib.request

# Retrieve a single page and report the URL and contents
def request_turbo_server(filename):
    if filename.endswith('.xml'):
        req = urllib.request.Request(url='http://127.0.0.1:5000/',
                                    data=open(filename, 'rb').read(),
                                    headers={'Content-Type': 'application/xml'})
    else:
        req = urllib.request.Request(url='http://127.0.0.1:5000/',
                                    data=open(filename, 'rb').read(),
                                    headers={'Content-Type': 'text/plain'})
    with urllib.request.urlopen(req, timeout=300) as conn:
        return conn.read()



def find_DUC_files(dataset_path):
    files_to_process = []

    # Clusters
    for corpus in os.listdir(dataset_path):
        if corpus.startswith('DUC'):
            corpus_path = os.path.join(dataset_path, corpus)
            for cluster in os.listdir(corpus_path):
                cluster_path = os.path.join(corpus_path, cluster)
                if os.path.isdir(cluster_path):
                    for filename in os.listdir(os.path.join(cluster_path, 'docs')):
                        if filename.endswith('.xml') and not os.path.exists(os.path.join(cluster_path, 'docs', filename + '.conll')):
                            files_to_process.append(os.path.join(cluster_path, 'docs', filename))
                    for filename in os.listdir(os.path.join(cluster_path, 'models')):
                        if filename.endswith('.xml') and not os.path.exists(os.path.join(cluster_path, 'models', filename + '.conll')):
                            files_to_process.append(os.path.join(cluster_path, 'models', filename))

    return files_to_process

input_path = '/home/ppb/Data/summarization/datasets/'
files_to_process = find_DUC_files(input_path)
print(files_to_process)

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    # Start the load operations and mark each future with its URL
    futures = {executor.submit(request_turbo_server, filename): filename for filename in files_to_process}
    for future in concurrent.futures.as_completed(futures):
        filename = futures[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (filename, exc))
        else:
            with open(filename+'.conll', 'w', encoding='utf8') as fp:
                fp.write(data.decode('utf8'))
                print('Saved file {}'.format(filename+'.conll'))