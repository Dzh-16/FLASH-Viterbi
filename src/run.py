import re, subprocess, os, csv
from datetime import datetime

base_path = './SIEVE/src/'
data_path = './SIEVE/src/data/'
result_path = './SIEVE/src/result/'
file_names = ['FLASH_Viterbi_multithread','FLASH_BS_Viterbi_multithread','vanilla Viterbi','checkpoint Viterbi','SIEVE-Mp','SIEVE-BS','SIEVE-BS-Mp']
parameters = [
    {
        'K_STATE': 3965,
        'T_STATE': 50,
        'obserRouteLEN': 256,
        'prob': 0.112,
        'MAX_THREADS': 1,
        'BeamSearchWidth': 32
    },
    {
        'K_STATE': 3965,
        'T_STATE': 50,
        'obserRouteLEN': 256,
        'prob': 0.169,
        'MAX_THREADS': 1,
        'BeamSearchWidth': 32
    },
]
def run_c(filename, p):
    with open(base_path + filename + '.c', 'r') as file:
        content = file.read()
    content = re.sub(r'#define K_STATE \d+', f"#define K_STATE {p['K_STATE']}", content)
    content = re.sub(r'#define T_STATE \d+', f"#define T_STATE {p['T_STATE']}", content)
    content = re.sub(r'#define obserRouteLEN \d+', f"#define obserRouteLEN {p['obserRouteLEN']}", content)
    content = re.sub(r'const float prob = \d+\.\d+;', f"const float prob = {p['prob']};", content)
    content = re.sub(r'const char data_path\[\] = "[^"]*";', f'const char data_path[] = "{data_path}";', content)
    if filename == 'FLASH_Viterbi_multithread' or filename == 'FLASH_BS_Viterbi_multithread':
        content = re.sub(r'#define MAX_THREADS \d+', f"#define MAX_THREADS {p['MAX_THREADS']}", content)
    if filename == 'FLASH_BS_Viterbi_multithread' or filename == 'SIEVE-BS' or filename == 'SIEVE-BS-Mp':
        content = re.sub(r'const int BeamSearchWidth = \d+;', f"const int BeamSearchWidth = {p['BeamSearchWidth']};", content)
    
    def get_decimal_places(num):
        num_str = str(num)
        if '.' in num_str:
            decimal_part = num_str.split('.')[1]
            return len(decimal_part)
        else:
            return 0
    problen = get_decimal_places(p['prob'])
    content = re.sub(r'prob%\.\d+f', f'prob%.{problen}f', content)

    
    modified_filename = f'{filename}_modified'
    with open(base_path + modified_filename + '.c', 'w') as file:
        file.write(content)

    compile_command = ['gcc', '-g', '-pthread', base_path + modified_filename + '.c', '-o', base_path + modified_filename, '-lm', '-Wl,-z,stack-size=268435456']
    if filename == 'SIEVE-BS' or filename == 'SIEVE-BS-Mp':
        pkg_config = subprocess.run(["pkg-config", "--cflags", "--libs", "glib-2.0"],capture_output=True, text=True, check=True)
        pkg_flags = pkg_config.stdout.strip().split()
        o_index = compile_command.index('-o')
        for flag in reversed(pkg_flags):
            compile_command.insert(o_index, flag)

    compile_result = subprocess.run(compile_command, capture_output=True, text=True)
    
    if compile_result.returncode != 0:
        print(f"compile ERROR: {compile_result.stderr}")
        return None
    
    run_result = subprocess.run([base_path + modified_filename], capture_output=True, text=True)
    
    if run_result.returncode != 0:
        print(f"run ERROR: {run_result.stderr}")
        return None
    
    output = run_result.stdout
    outputTime = re.search(r'time: ([\d.]+)', output).group(1)
    outputMemory = re.search(r'memory: (\d+)', output).group(1)
    print(f"{filename} Time: {outputTime}, Memory: {outputMemory}")
    return outputTime, outputMemory

def run_result(filename, p, writer, csv_file):
    outtime, outmemory = run_c(filename, p)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp,
        p['K_STATE'],
        p['T_STATE'],
        p['obserRouteLEN'],
        p['prob'],
        p.get('MAX_THREADS', 'N/A'),
        p.get('BeamSearchWidth', 'N/A'),
        outtime,
        outmemory]
    writer.writerow(row)
    csv_file.flush()

def main():
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for filename in file_names:
        csv_filename = f"{filename}_result"
        file_exists = os.path.exists(result_path + csv_filename + '.csv')
        with open(result_path + csv_filename + '.csv', 'a', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(['timestamp', 'K_STATE', 'T_STATE', 'obserRouteLEN', 'prob', 'MAX_THREADS', 'BeamSearchWidth', 'time', 'memory'])
            for p in parameters:
                run_result(filename,p,writer,csv_file)

if __name__ == '__main__':
    main()