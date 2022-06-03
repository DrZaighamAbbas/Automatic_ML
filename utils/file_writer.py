def write_string_to_file(file_name, string_content):
    write_to_file(file_name, string_content, mode="w")


def write_to_file(file_name, string_content, mode = "wb"):
    with open(file_name, mode) as f:
        f.write(string_content)
