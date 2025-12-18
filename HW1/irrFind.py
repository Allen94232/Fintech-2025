import numpy as np
from scipy.optimize import fsolve, brentq

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    """
    Calculate Internal Rate of Return (IRR) for given cash flows.
    Optimized version with better efficiency and numerical stability.
    
    Args:
        cashFlowVec: List of cash flows
        cashFlowPeriod: Period (in months) for cash flow
        compoundPeriod: Period (in months) for compounding
    
    Returns:
        IRR as a decimal (e.g., 0.05 for 5%), or 0 if input is invalid
    """
    
    # Input validation
    try:
        if not cashFlowVec or len(cashFlowVec) < 2:
            return 0.0
        
        if not isinstance(cashFlowPeriod, (int, float)) or cashFlowPeriod <= 0:
            return 0.0
        
        if not isinstance(compoundPeriod, (int, float)) or compoundPeriod <= 0:
            return 0.0
        
        if cashFlowPeriod % compoundPeriod != 0:
            return 0.0
        
        for cf in cashFlowVec:
            if not isinstance(cf, (int, float)) or not np.isfinite(cf):
                return 0.0
        
        if all(cf == 0 for cf in cashFlowVec):
            return 0.0
        
        has_positive = any(cf > 0 for cf in cashFlowVec)
        has_negative = any(cf < 0 for cf in cashFlowVec)
        if not (has_positive and has_negative):
            return 0.0
            
    except Exception:
        return 0.0
    
    def npv_equation(annual_rate):
        """Calculate Net Present Value equation that should equal zero at IRR"""
        if annual_rate <= -1:
            return float('inf')
        
        npv_value = 0.0
        
        for i, cf in enumerate(cashFlowVec):
            if i == 0:
                npv_value += cf
            else:
                time_in_years = (i * cashFlowPeriod) / 12.0
                compounding_frequency = 12.0 / compoundPeriod
                
                if compoundPeriod == 12:
                    discount_factor = (1 + annual_rate) ** time_in_years
                else:
                    discount_factor = (1 + annual_rate / compounding_frequency) ** (compounding_frequency * time_in_years)
                
                pv = cf / discount_factor
                npv_value += pv
        
        return npv_value
    
    try:
        # Strategy 1: Use limited, high-quality initial guesses with relaxed tolerance
        best_solutions = []
        
        # Priority guesses - most likely to succeed
        priority_guesses = [0.0, 0.05, -0.05, 0.02, -0.02]
        
        for guess in priority_guesses:
            try:
                solution = fsolve(npv_equation, guess, full_output=True)
                irr_candidate = solution[0][0]
                residual = abs(solution[1]['fvec'][0])
                
                # Relaxed tolerance from 1e-8 to 1e-6
                if residual < 1e-6:
                    best_solutions.append(irr_candidate)
                    
            except:
                continue
        
        # If found solutions within -10% to 10% range, return immediately
        if best_solutions:
            valid_solutions = [sol for sol in best_solutions if -0.1 <= sol <= 0.1]
            
            if valid_solutions:
                valid_solutions.sort()
                positive_solutions = [sol for sol in valid_solutions if sol > 0]
                if positive_solutions:
                    return min(positive_solutions)
                else:
                    return max(valid_solutions)
        
        # Strategy 2: Extended search with fewer guesses
        extended_guesses = [0.01, -0.01, 0.08, -0.08, 0.03, -0.03]
        
        for guess in extended_guesses:
            try:
                solution = fsolve(npv_equation, guess, full_output=True)
                irr_candidate = solution[0][0]
                residual = abs(solution[1]['fvec'][0])
                
                if residual < 1e-6 and -0.1 <= irr_candidate <= 0.1:
                    return irr_candidate
                    
            except:
                continue
        
        # Strategy 3: Fallback - interval scanning + bisection method
        try:
            search_points = np.linspace(-0.09, 0.09, 50)
            
            for i in range(len(search_points) - 1):
                try:
                    npv1 = npv_equation(search_points[i])
                    npv2 = npv_equation(search_points[i + 1])
                    
                    # Check for sign change (zero crossing)
                    if npv1 * npv2 < 0:
                        root = brentq(npv_equation, search_points[i], search_points[i + 1])
                        if -0.1 <= root <= 0.1:
                            return root
                except:
                    continue
        except:
            pass
        
        return 0.0
        
    except Exception:
        return 0.0

