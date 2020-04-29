from lib import get_data

if __name__ == '__main__':
    model_df = get_data.get_model_data(date_range=(0, 14), pred_day=21)

    # for i in [(7, 14), (14, 21), (14, 16)]:
    #     print(len(get_model_data(date_range=(0, i[0]), pred_day=i[1])))

    # 1187
    # 750
    # 1061
