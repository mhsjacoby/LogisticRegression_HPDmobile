

from etl import ETL

class test_t(ETL):
    def __init__(self, fill_type,  H_num='H1', hub='RS4'):
        super().__init__(H_num='H1', hub='RS4')
        self.train_configs = self.read_config(config_type='train')
        # pass


    # def __init__(H_num, hub):

if __name__=='__main__':
    t = test_t(fill_type='ones')
    # print(t.fill_type)
    # print(t.days)
    # for x in dir(t):
    #     print(x)
    # print(dir(t))
    # print(t.configs)