#例外処理
def divide(a, b):
    try:
        result = a / b # 例外が発生する可能性のあるコード（例外が絶対に発生しないコードを含むべきではない⇒なるべく短く）
    except ZeroDivisionError:
        return print("Error: Division by zero is not allowed.")
    except TypeError:
        return print("Error: Invalid input type. Please provide numbers.")
    else: # 例外が発生しなかった場合に実行されるコード
        return print(result)
    finally: # 例外の有無にかかわらず、必ず実行されるコード
        print("Execution completed.")   

divide(10, 2)  # 正常な除算
divide(10, 0)  # ゼロ除算
divide(10, 'a')  # 無効な入力タイプ