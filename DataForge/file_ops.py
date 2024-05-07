def write_text_to_file(file_name, text):
    with open(file_name, 'w') as file:
        file.write(text)

    @staticmethod
    def read_text_from_file(file_name):
        with open(file_name, 'r') as file:
            return file.read()
        
