import sys
from irrFind import irrFind

def main():
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                print("0.0000")
                continue
            
            # 依題目輸入，前 n 個是 cash flows，最後兩個是 periods
            try:
                nums = [int(x) for x in parts]
            except ValueError:
                try:
                    # 若有非整數，嘗試浮點解析
                    nums = [float(x) for x in parts]
                except ValueError:
                    # 如果還是無法解析，輸出 0
                    print("0.0000")
                    continue

            if len(nums) < 3:
                print("0.0000")
                continue

            cash_flows = nums[:-2]
            cash_flow_period = nums[-2]
            compound_period = nums[-1]

            irr = irrFind(cash_flows, cash_flow_period, compound_period)
            if irr is None or irr == 0:
                # 若沒找到合格解或返回 0，輸出 0.0000
                print("0.0000")
            else:
                # 輸出百分比，四位小數（符合 judge 規格）
                print(f"{irr * 100:.4f}")
                
        except Exception:
            # 捕獲任何其他異常，輸出 0
            print("0.0000")

if __name__ == "__main__":
    main()
