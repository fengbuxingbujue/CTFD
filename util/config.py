import yaml
import os


class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
            # 注意这里的代码，在读取文件是选择的模式是'r'，是只读模式
            # self._dict = yaml.safe_load(self._yaml) 这句代码将yml文件中的内容读取到变量self._dict中
            # 此时self._dict的类型为字典
            # self._dict['PATH'] = os.path.dirname(config_path) 后一句代码是对self._dict进行修改，
            # 向字典中添加了新的一项 'PATH': os.path.dirname(config_path)，代码内存中的字典多了一项，
            # 但原本yml文件不会发生变化

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


def load_config(path):
    config_path = path
    config = Config(config_path)
    return config


'''
if __name__ == "__main__":
    config = load_config(f'D:/pycharm/pythonPytochProject1/DocDiff_main/conf.yml')
    config.print()
    print(config._dict)
'''