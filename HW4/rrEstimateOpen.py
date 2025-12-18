import sys
import numpy as np
import pandas as pd
from myAction import *
import time
import copy

def computeReturnRate(priceMat, transFeeRate1, transFeeRate2, actionMat, K, problem_type):
    capital = 1000       # Initial available capital
    capitalOrig = capital
    stockCount = len(priceMat[0])

    if len(actionMat) > K:
        print("truncate")
        actionMat = actionMat[:K]   # truncate extra trades

    if len(actionMat) == 0:
        returnRate = 0
        return returnRate
    
    suggestedAction = actionMat
    actionCount = len(suggestedAction)
    
    stockHolding = np.zeros((actionCount, stockCount))
    realAction = np.zeros((actionCount, 1))
    preDay = -10

    # Run through each action, should order by day
    for i in range(actionCount):
        actionVec = actionMat[i]
        day = actionVec[0]
        a = actionVec[1]
        b = actionVec[2]
        z = actionVec[3]
        currentPriceVec = priceMat[day]

        try:
            # Enforce cooldown for all problems
            if day < preDay + 3 or z <= 0:
                raise AssertionError("Cooldown violation or zero amount")

            # Copy previous stock holding
            stockHolding[i] = stockHolding[i - 1] if i > 0 else np.zeros(stockCount)

            preDay = day  # update last action day

            if a == -1 and b >= 0 and capital > 0:   # buy
                currentPrice = currentPriceVec[b]
                if capital < z:
                    z = capital
                stockHolding[i][b] += z * (1 - transFeeRate1) / currentPrice
                capital = capital - z
                realAction[i] = 1

            elif b == -1 and a >= 0 and stockHolding[i][a] > 0:  # sell
                currentPrice = currentPriceVec[a]
                sellStock = z / currentPrice
                if stockHolding[i][a] < sellStock:
                    sellStock = stockHolding[i][a]
                getCash = sellStock * currentPrice * (1 - transFeeRate2)
                capital = capital + getCash
                stockHolding[i][a] -= sellStock
                realAction[i] = -1

            else:
                raise AssertionError("Invalid action: selling stock you don't have or wrong format")

        except AssertionError as e:
            print(f"Action {i} failed: {e}")
            raise  

    # calculate total cash you get at last day
    total = capital
    for stock in range(stockCount):
        currentPriceVec = priceMat[-1]
        total += stockHolding[-1][stock] * currentPriceVec[stock] * (1 - transFeeRate2)

    returnRate = (total - capitalOrig) / capitalOrig
    return returnRate



if __name__ == "__main__":

    print("Reading %s..." % (sys.argv[1]))
    file = sys.argv[1]
    df = pd.read_csv(file, delimiter=' ', header=None)
    transFeeRate1 = float(sys.argv[2])
    transFeeRate2 = float(sys.argv[3])
    priceMat = df.values

    problem_type = 1
    print("------------Problem 1-------------")
    start = time.time()
    # actionMat = myAction01(priceMat, transFeeRate1, transFeeRate2)
    actionMat = myAction01_Sample(priceMat, transFeeRate1, transFeeRate2)
    rr = computeReturnRate(priceMat, transFeeRate1, transFeeRate2, actionMat, 99999999, problem_type)
    end = time.time()
    print("Time:", end - start)
    print("rr=%f%%" % (rr * 100))

    K_list = [100, 150, 200]
    problem_type = 2
    print("------------Problem 2-------------")
    start = time.time()
    total_rr = 0
    for K in K_list:
        actionMat = myAction02(priceMat, transFeeRate1, transFeeRate2, K)
        rr = computeReturnRate(priceMat, transFeeRate1, transFeeRate2, actionMat, K, problem_type)
        total_rr += rr

    end = time.time()
    print("Time:", end - start)
    print("rr=%f%%" % (total_rr * 100 / 3))

    print("------------Problem 3-------------")
    start = time.time()

    K = 999999          # unlimited transactions
    actions = []        
    position = np.zeros(priceMat.shape[1] + 1)
    position[-1] = 1000   # initial cash
    actionHistory = []

    for day in range(len(priceMat)):
        priceMatHistory = priceMat[:day+1]

        if day == len(priceMat) - 1:
            priceMatFuture = np.array([])   
        else:
            priceMatFuture = priceMat[day+1:day+2]   

        # action = myAction03(
        #     priceMatHistory=priceMatHistory,
        #     priceMatFuture=priceMatFuture,
        #     position=position,
        #     actionHistory=actionHistory,
        #     rate1=transFeeRate1,
        #     rate2=transFeeRate2
        # )

        action = myAction03_Sample(
            priceMatHistory=priceMatHistory,
            priceMatFuture=priceMatFuture,
            position=position,
            actionHistory=actionHistory,
            rate1=transFeeRate1,
            rate2=transFeeRate2
        )

        if action is not None:
            # ensure day, a, b are integers, z is float
            action_fixed = [int(action[0]), int(action[1]), int(action[2]), float(action[3])]
            actions.append(action_fixed)
            actionHistory.append(action_fixed)

            day_p, a, b, z = action_fixed
            todayPrice = priceMat[day]

            if z > 0:
                if a == -1 and b >= 0:      # buy stock b
                    idx = b
                    cost = min(z, position[-1])
                    shares = cost * (1 - transFeeRate1) / todayPrice[idx]
                    position[idx] += shares
                    position[-1] -= cost

                elif b == -1 and a >= 0:    # sell stock a
                    idx = a
                    shares_to_sell = min(z / todayPrice[idx], position[idx])
                    cash_gain = shares_to_sell * todayPrice[idx] * (1 - transFeeRate2)
                    position[idx] -= shares_to_sell
                    position[-1] += cash_gain

    if actions:
        actionMat = np.array(actions, dtype=object)  
    else:
        actionMat = np.zeros((0, 4), dtype=float)  

    # calculate total return
    rr = computeReturnRate(priceMat, transFeeRate1, transFeeRate2, actionMat, K, problem_type=3)

    end = time.time()
    print("Time:", end - start)
    print("rr=%f%%" % (rr * 100))
