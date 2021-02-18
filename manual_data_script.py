ALIGNED_FILE_PATH = ".\\sentence-aligned.v2\\simple.aligned"

# read .aligned file type
def read_aligned_file(file_name):
    with open(file_name, "r", encoding='utf-8', errors='ignore') as f:
        text_lines = f.readlines()[:20]
        for i, line in enumerate(text_lines):
            #!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
            start = line.find("\t")
            start = line.find("\t", start + 1) + 1
            line = line[start:]
            line = line.replace(" . ", ". ")
            line = line.replace(" .\n", ".\n")
            line = line.replace(" , ", ", ")
            line = line.replace(" -", "-")
            line = line.replace("- ", "-")
            line = line.replace(" '", "'")
            line = line.replace(" :", ":")
            line = line.replace(" ;", ";")
            line = line.replace("` ", "`")
            print(line)

if __name__ == "__main__":
    sentences = read_aligned_file(ALIGNED_FILE_PATH)

