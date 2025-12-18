import numpy as np

# =========================================================
#  Algorithmic Trading Optimization Module
#  Author: [Persona Expert]
#  Context: Optimized for 4-stock portfolio with 3-day constraints
# =========================================================

# --- Constants ---
BUY_FEE = 0.001425
SELL_FEE = 0.004425
COOLDOWN = 3  # Gap between trades (Action at t, next at t+3)

def solve_global_dp(priceMat, rate1, rate2):
    """
    Solves Problem 1 (Unconstrained) using Forward DP.
    
    State: dp[t, s] = Max Wealth at end of day t holding asset s.
    s = 0 (Cash), 1..N (Stocks)
    """
    n_days, n_stocks = priceMat.shape
    n_states = n_stocks + 1
    
    # DP Table: -> Value
    dp = np.full((n_days, n_states), -1.0)
    
    # Backtracking Tables
    # parent_state: Which state did we come from?
    # action_type: 0=Hold, 1=Buy, 2=Sell
    parent_state = np.zeros((n_days, n_states), dtype=int)
    action_type = np.zeros((n_days, n_states), dtype=int)
    
    # --- Initialization (Day 0) ---
    INITIAL_CAPITAL = 1000.0
    
    # State 0: Cash
    dp[0, 0] = INITIAL_CAPITAL 
    parent_state[0, 0] = 0
    action_type[0, 0] = 0
    
    # States 1..N: Stocks (Immediate Buy)
    for s in range(n_stocks):
        # Wealth = Cash * (1 - fee)
        # Note: We track "Current Market Value" of holdings
        dp[0, s+1] = INITIAL_CAPITAL * (1 - rate1)
        parent_state[0, s+1] = 0 # From Cash
        action_type[0, s+1] = 1  # Buy
        
    # --- Forward Pass ---
    for t in range(1, n_days):
        # 1. Passive Evolution (Hold)
        # ---------------------------
        
        # Cash: No change (assuming 0 interest)
        dp[t, 0] = dp[t-1, 0]
        parent_state[t, 0] = 0
        action_type[t, 0] = 0 # Hold
        
        # Stocks: Price evolution
        current_prices = priceMat[t]
        prev_prices = priceMat[t-1]
        price_ratios = current_prices / prev_prices
        
        for s in range(n_stocks):
            prev_val = dp[t-1, s+1]
            if prev_val > 0:
                new_val = prev_val * price_ratios[s]
                dp[t, s+1] = new_val
                parent_state[t, s+1] = s+1
                action_type[t, s+1] = 0 # Hold

        # 2. Active Transitions (Check Cooldown)
        # --------------------------------------
        prev_act_t = t - COOLDOWN
        
        if prev_act_t >= 0:
            # A) Try SELLING (Stock -> Cash)
            for s in range(n_stocks):
                # Look back to state at prev_act_t
                # Note: We act at t. The "holding" during gap is passive.
                # So we take Value at prev_act_t, evolve it to t, then sell.
                
                # Correction: The DP state at prev_act_t stores the optimal value.
                # We project that value to today's price.
                val_at_prev = dp[prev_act_t, s+1]
                
                if val_at_prev > 0:
                    # Calculate Market Value at day t
                    p_ratio_gap = priceMat[t, s] / priceMat[prev_act_t, s]
                    mkt_val_t = val_at_prev * p_ratio_gap
                    
                    # Apply Sell Fee
                    cash_proceeds = mkt_val_t * (1 - rate2)
                    
                    # Update if better than holding cash
                    if cash_proceeds > dp[t, 0]:
                        dp[t, 0] = cash_proceeds
                        parent_state[t, 0] = s+1 # Came from Stock s
                        action_type[t, 0] = 2    # Sell
            
            # B) Try BUYING (Cash -> Stock)
            cash_avail = dp[prev_act_t, 0]
            if cash_avail > 0:
                for s in range(n_stocks):
                    # Buy Stock s
                    # Market Value = Cash * (1 - rate1)
                    buy_val = cash_avail * (1 - rate1)
                    
                    # Update if better than holding stock
                    if buy_val > dp[t, s+1]:
                        dp[t, s+1] = buy_val
                        parent_state[t, s+1] = 0 # Came from Cash
                        action_type[t, s+1] = 1  # Buy

    # --- Backtracking (Path Reconstruction) ---
    actions = []
    
    # Start from best state at final day
    curr_state = np.argmax(dp[n_days-1])
    curr_t = n_days - 1
    
    # If final state is stock, force sell? No, maximize "Market Value" is the goal.
    # The instructions say "performance... calculated as cash equivalent".
    # DP naturally maximizes this.
    
    while curr_t >= 0:
        act = action_type[curr_t, curr_state]
        prev_s = parent_state[curr_t, curr_state]
        
        if act == 0: # Hold
            curr_t -= 1
            # State remains same (logically), just index scan back
            
        elif act == 1: # Buy (Cash -> Stock)
            # Transaction occurred at curr_t
            # Previous valid decision point was prev_act_t
            stock_idx = curr_state - 1
            prev_t = curr_t - COOLDOWN
            if prev_t < 0: prev_t = -1 
            
            # Z = Cash amount used
            z_val = dp[prev_t, 0] if prev_t >= 0 else INITIAL_CAPITAL
            
            actions.append([curr_t, -1, stock_idx, z_val])
            
            curr_state = 0 # We came from Cash
            curr_t = prev_t # Jump
            
        elif act == 2: # Sell (Stock -> Cash)
            stock_idx = prev_s - 1
            prev_t = curr_t - COOLDOWN
            
            # Z = Market Value of stock sold
            # Reconstruct value from previous state
            val_at_prev = dp[prev_t, prev_s]
            p_ratio = priceMat[curr_t, stock_idx] / priceMat[prev_t, stock_idx]
            z_val = val_at_prev * p_ratio
            
            actions.append([curr_t, stock_idx, -1, z_val])
            
            curr_state = prev_s
            curr_t = prev_t
            
    actions.reverse()
    return actions

def solve_limited_dp(priceMat, rate1, rate2, K):
    """
    Solves Problem 2 (K Transactions) using Layered DP.
    """
    n_days, n_stocks = priceMat.shape
    n_states = n_stocks + 1
    
    # DP: [k, t, s]
    # To save memory, we only need 'current k' and 'previous k'
    # But for backtracking, we need full history.
    # Given memory limits, we can keep full table if K*N*T is small.
    # 400 * 1000 * 5 * 8 bytes ~= 16MB. Safe.
    
    dp = np.full((K+1, n_days, n_states), -1.0)
    # parent pointers must track (t_prev, s_prev)
    # We implicitly know k_prev is k for Hold, k-1 for Action.
    
    # Store actions directly to simplify backtracking:
    # record[k, t, s] = (action_code, prev_s_idx)
    record = np.zeros((K+1, n_days, n_states, 2), dtype=int)
    
    INITIAL_CAPITAL = 1000.0
    
    # k=0 Layer: Always Cash, or Initial Holdings if allowed (not allowed, start cash)
    dp[:, 0, 0] = INITIAL_CAPITAL
    
    # Initialize k=1 layer (First Buy) at Day 0
    # Note: Buying at Day 0 counts as 1 transaction.
    for s in range(n_stocks):
        dp[1, 0, s+1] = INITIAL_CAPITAL * (1 - rate1)
        record[1, 0, s+1, :] = [1, 0]  # Buy from Cash
    
    # Forward Pass
    for k in range(1, K+1):
        for t in range(1, n_days):
            # 1. Passive Hold (from same k)
            # Cash
            if dp[k, t-1, 0] > dp[k, t, 0]:
                dp[k, t, 0] = dp[k, t-1, 0]
                record[k, t, 0, :] = [0, 0]  # Hold
                
            # Stocks
            p_ratios = priceMat[t] / priceMat[t-1]
            for s in range(n_stocks):
                prev_val = dp[k, t-1, s+1]
                if prev_val > 0:
                    new_val = prev_val * p_ratios[s]
                    if new_val > dp[k, t, s+1]:
                        dp[k, t, s+1] = new_val
                        record[k, t, s+1, :] = [0, s+1] # Hold
                        
            # 2. Active Transitions (from k-1)
            # Only if cooldown allows
            prev_act_t = t - COOLDOWN
            if prev_act_t >= 0:
                # Try Buy (from Cash at k-1)
                cash_prev = dp[k-1, prev_act_t, 0]
                if cash_prev > 0:
                    for s in range(n_stocks):
                        buy_val = cash_prev * (1 - rate1)
                        if buy_val > dp[k, t, s+1]:
                            dp[k, t, s+1] = buy_val
                            record[k, t, s+1, :] = [1, 0]  # Buy (new trans)
                            
                # Try Sell (from Stock at k-1)
                for s in range(n_stocks):
                    stock_prev = dp[k-1, prev_act_t, s+1]
                    if stock_prev > 0:
                        p_ratio_gap = priceMat[t, s] / priceMat[prev_act_t, s]
                        mkt_val = stock_prev * p_ratio_gap
                        sell_val = mkt_val * (1 - rate2)
                        
                        if sell_val > dp[k, t, 0]:
                            dp[k, t, 0] = sell_val
                            record[k, t, 0, :] = [2, s+1] # Sell (new trans)

    # Find best K and best State
    # It's possible using fewer than K transactions is optimal
    # We check all k at T-1
    best_k, best_s = np.unravel_index(np.argmax(dp[:, n_days-1, :]), dp[:, n_days-1, :].shape)
    
    # Backtrack
    actions = []
    curr_k = best_k
    curr_t = n_days - 1
    curr_s = best_s
    
    while curr_t >= 0 and curr_k > 0:
        act_code, prev_s_idx = record[curr_k, curr_t, curr_s]
        
        if act_code == 0: # Hold
            curr_t -= 1
        elif act_code == 1: # Buy
            # Consumed 1 transaction
            stock_idx = curr_s - 1
            prev_t = curr_t - COOLDOWN
            if prev_t < 0: prev_t = -1
            
            z_val = dp[curr_k-1, prev_t, 0] if prev_t >= 0 else INITIAL_CAPITAL
            actions.append([curr_t, -1, stock_idx, z_val])
            
            curr_k -= 1
            curr_t = prev_t
            curr_s = 0
        elif act_code == 2: # Sell
            stock_idx = prev_s_idx - 1
            prev_t = curr_t - COOLDOWN
            
            val_prev = dp[curr_k-1, prev_t, prev_s_idx]
            p_ratio = priceMat[curr_t, stock_idx] / priceMat[prev_t, stock_idx]
            z_val = val_prev * p_ratio
            
            actions.append([curr_t, stock_idx, -1, z_val])
            
            curr_k -= 1
            curr_t = prev_t
            curr_s = prev_s_idx
            
    actions.reverse()
    return actions

# --- Main Interface Functions ---

def myAction01(priceMat, rate1, rate2):
    """Problem 1: Unconstrained optimal trading"""
    if not isinstance(priceMat, np.ndarray):
        priceMat = np.array(priceMat)
    return solve_global_dp(priceMat, rate1, rate2)

def myAction02(priceMat, rate1, rate2, K):
    """Problem 2: K-constrained trading"""
    if not isinstance(priceMat, np.ndarray):
        priceMat = np.array(priceMat)
    return solve_limited_dp(priceMat, rate1, rate2, K)

def myAction03(priceMatHistory, priceMatFuture, position, actionHistory, rate1, rate2):
    """
    Problem 3: Online mode trading - 優化版本
    
    策略：使用與 myAction01_Sample 相同的貪婪邏輯
    1. 優先賣出明天會跌的股票（選跌幅最大的）
    2. 買入明天會漲且扣除手續費後仍有利潤的股票
    """
    EPSILON = 1e-10
    
    if not isinstance(priceMatHistory, np.ndarray):
        priceMatHistory = np.array(priceMatHistory)
    
    stockCount = priceMatHistory.shape[1]
    day_p = priceMatHistory.shape[0] - 1
    todayPrices = priceMatHistory[-1]
    holdings = position[:-1]
    cash = position[-1]
    
    # 檢查冷卻期
    if len(actionHistory) > 0:
        last_action_day = actionHistory[-1][0]
        if day_p < last_action_day + COOLDOWN:
            return None
    
    # 最後一天：賣出所有持股
    if priceMatFuture.size == 0:
        for i in range(stockCount):
            if holdings[i] > EPSILON:
                sell_amount = holdings[i] * todayPrices[i]
                return np.array([day_p, i, -1, sell_amount])
        return None
    
    nextPrices = priceMatFuture[0]
    
    # 使用與 myAction01_Sample 相同的貪婪邏輯
    max_score = 1.0
    best_action = None
    
    # 1. 優先考慮賣出：找出明天會下跌最多的持股
    best_sell_idx = -1
    for i in range(stockCount):
        if holdings[i] > EPSILON and todayPrices[i] > 0:
            if nextPrices[i] > 0:
                # 計算「今天價格/明天價格」，越大表示明天跌越多
                loss_avoidance_score = todayPrices[i] / nextPrices[i]
            else:
                loss_avoidance_score = float('inf')
            
            if loss_avoidance_score > max_score:
                max_score = loss_avoidance_score
                best_sell_idx = i
    
    # 2. 考慮買入：找出明天漲幅最大且有利可圖的股票
    best_buy_idx = -1
    if cash > EPSILON:
        for j in range(stockCount):
            if todayPrices[j] > 0:
                # 計算扣除手續費後的實際收益率
                profit_ratio = (nextPrices[j] * (1.0 - rate2)) / (todayPrices[j] * (1.0 + rate1))
            else:
                profit_ratio = 0
            
            if profit_ratio > max_score:
                max_score = profit_ratio
                best_buy_idx = j
    
    # 3. 執行最佳動作（賣出優先於買入）
    if best_sell_idx != -1 and todayPrices[best_sell_idx] / nextPrices[best_sell_idx] == max_score:
        sell_amount = holdings[best_sell_idx] * todayPrices[best_sell_idx]
        if sell_amount > EPSILON:
            return np.array([day_p, best_sell_idx, -1, sell_amount])
    
    if best_buy_idx != -1 and max_score > 1.0:
        return np.array([day_p, -1, best_buy_idx, cash])
    
    return None


# --- Sample Functions (for reference) ---

def myAction01_Sample(priceMat, rate1, rate2):
    """範例函數：貪婪策略"""
    cash = 1000.0
    nextDay = 1
    dataLen, stockCount = priceMat.shape

    stockHolding = np.zeros((dataLen, stockCount))
    actionMat = []
    cooldownEndDay = 0

    for day in range(0, dataLen - nextDay):
        dayPrices = priceMat[day]
        nextDayPrices = priceMat[day + nextDay]

        if day > 0:
            stockHolding[day] = stockHolding[day-1]

        if day < cooldownEndDay:
            continue

        best_action = None
        max_score = 1.0
        best_sell_idx = -1

        for i in range(stockCount):
            holding = stockHolding[day][i]
            if holding > 0:
                if nextDayPrices[i] > 0:
                    loss_avoidance_score = dayPrices[i] / nextDayPrices[i]
                else:
                    loss_avoidance_score = 0

                if loss_avoidance_score > max_score:
                    max_score = loss_avoidance_score
                    best_sell_idx = i

        best_buy_idx = -1

        if cash > 0:
            for j in range(stockCount):
                if dayPrices[j] > 0:
                    profit_ratio = (nextDayPrices[j] * (1.0 - rate2)) / (dayPrices[j] * (1.0 + rate1))
                else:
                    profit_ratio = 0

                if profit_ratio > max_score:
                    max_score = profit_ratio
                    best_buy_idx = j

        if best_sell_idx != -1 and dayPrices[best_sell_idx] / nextDayPrices[best_sell_idx] == max_score:
            sell_units = stockHolding[day][best_sell_idx]
            sell_amount = sell_units * dayPrices[best_sell_idx]
            cash += sell_units * dayPrices[best_sell_idx] * (1.0 - rate2)
            stockHolding[day][best_sell_idx] = 0.0
            best_action = [day, best_sell_idx, -1, sell_amount]
            cooldownEndDay = day + 3

        elif best_buy_idx != -1 and max_score > 1.0:
            buy_amount = cash
            units_to_buy = buy_amount * (1.0 - rate1) / dayPrices[best_buy_idx]
            stockHolding[day][best_buy_idx] += units_to_buy
            cash = 0.0
            best_action = [day, -1, best_buy_idx, buy_amount]
            cooldownEndDay = day + 3

        if best_action is not None:
            actionMat.append(best_action)

    return actionMat


def myAction03_Sample(priceMatHistory, priceMatFuture, position, actionHistory, rate1, rate2):
    """範例函數：線上模式貪婪策略"""
    EPSILON = 1e-10
    
    stockCount = priceMatHistory.shape[1]
    day_p = priceMatHistory.shape[0] - 1
    todayPrices = priceMatHistory[-1]
    holdings = position[:-1]
    cash = position[-1]

    if len(actionHistory) != 0:
        if day_p < actionHistory[-1][0] + 3:
            return None

    if priceMatFuture.size == 0:
        for i in range(stockCount):
            if holdings[i] > EPSILON:
                sell_amount = holdings[i] * todayPrices[i]
                return np.array([day_p, i, -1, sell_amount])
        return None

    nextDayPrices = priceMatFuture[0]

    max_loss_ratio = 1.0
    best_sell_stock_idx = -1
    for i in range(stockCount):
        if holdings[i] > EPSILON:
            if nextDayPrices[i] > 0:
                loss_ratio = todayPrices[i] / nextDayPrices[i]
            else:
                loss_ratio = 0

            if loss_ratio > max_loss_ratio:
                max_loss_ratio = loss_ratio
                best_sell_stock_idx = i

    if best_sell_stock_idx != -1:
        sell_amount = holdings[best_sell_stock_idx] * todayPrices[best_sell_stock_idx]
        if sell_amount > EPSILON:
            return np.array([day_p, best_sell_stock_idx, -1, sell_amount])
        else:
            best_sell_stock_idx = -1

    if cash > EPSILON:
        max_profit_ratio = 1.0
        best_buy_stock_idx = -1

        for j in range(stockCount):
            if todayPrices[j] > 0:
                profit_ratio = (nextDayPrices[j] * (1.0 - rate2)) / (todayPrices[j] * (1.0 + rate1))
            else:
                profit_ratio = 0

            if profit_ratio > max_profit_ratio:
                max_profit_ratio = profit_ratio
                best_buy_stock_idx = j

        if best_buy_stock_idx != -1:
            buy_amount = cash
            return np.array([day_p, -1, best_buy_stock_idx, buy_amount])

    return None