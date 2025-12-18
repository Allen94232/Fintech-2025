#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""
    # 4G = 2(2G)
    # First double: 2G
    twoG = G + G
    # Second double: 2(2G) = 4G
    fourG = twoG + twoG
    return fourG


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""
    # 5G = 4G + G
    # First compute 4G
    twoG = G + G
    fourG = twoG + twoG
    # Then add G
    fiveG = fourG + G
    return fiveG


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""
    
    if n == 0:
        return callback_get_INFINITY(), 0, 0
    
    if n == 1:
        return point, 0, 0
    
    # Convert n to binary (without '0b' prefix)
    binary = bin(n)[2:]
    
    # Initialize result with the point (corresponding to the leftmost 1 in binary)
    result = point
    num_doubles = 0
    num_additions = 0
    
    # Process remaining bits from left to right
    for bit in binary[1:]:
        # Double
        result = result + result
        num_doubles += 1
        
        # Add if bit is 1
        if bit == '1':
            result = result + point
            num_additions += 1
    
    return result, num_doubles, num_additions


#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""
    
    if n == 0:
        return callback_get_INFINITY(), 0, 0
    
    if n == 1:
        return point, 0, 0
    
    # Convert to Non-Adjacent Form (NAF)
    # NAF is a signed binary representation where no two adjacent digits are non-zero
    naf = []
    while n > 0:
        if n % 2 == 1:  # n is odd
            naf_i = 2 - (n % 4)  # This gives us either 1 or -1
            n = (n - naf_i) // 2
        else:
            naf_i = 0
            n = n // 2
        naf.append(naf_i)
    
    # Reverse to process from most significant bit
    naf.reverse()
    
    # Remove leading zeros and find first non-zero
    while naf and naf[0] == 0:
        naf.pop(0)
    
    if not naf:
        return callback_get_INFINITY(), 0, 0
    
    # Initialize with first non-zero digit
    if naf[0] == 1:
        result = point
    else:  # naf[0] == -1
        result = -point
    
    num_doubles = 0
    num_additions = 0
    
    # Process remaining digits
    for i in range(1, len(naf)):
        # Double
        result = result + result
        num_doubles += 1
        
        # Add or subtract based on NAF digit
        if naf[i] == 1:
            result = result + point
            num_additions += 1
        elif naf[i] == -1:
            result = result + (-point)
            num_additions += 1
    
    return result, num_doubles, num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""
    
    G = callback_getG()
    n = callback_get_n()
    
    # Convert hash to integer
    z = int(hashID, 16)
    
    # Generate random k
    k = callback_randint(1, n - 1)
    
    # Calculate r = (k * G).x mod n
    kG = k * G
    r = kG.x() % n
    
    # If r == 0, we should regenerate k, but for simplicity we'll assume it won't be 0
    if r == 0:
        k = callback_randint(1, n - 1)
        kG = k * G
        r = kG.x() % n
    
    # Calculate s = k^(-1) * (z + r * private_key) mod n
    k_inv = pow(k, -1, n)  # Modular inverse of k
    s = (k_inv * (z + r * private_key)) % n
    
    # If s == 0, we should regenerate k, but for simplicity we'll assume it won't be 0
    if s == 0:
        k = callback_randint(1, n - 1)
        kG = k * G
        r = kG.x() % n
        k_inv = pow(k, -1, n)
        s = (k_inv * (z + r * private_key)) % n
    
    signature = (r, s)
    return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""
    
    G = callback_getG()
    n = callback_get_n()
    infinity_point = callback_get_INFINITY()
    
    r, s = signature
    
    # Check if r and s are in valid range
    if r < 1 or r >= n or s < 1 or s >= n:
        return False
    
    # Convert hash to integer
    z = int(hashID, 16)
    
    # Calculate w = s^(-1) mod n
    w = pow(s, -1, n)
    
    # Calculate u1 = z * w mod n
    u1 = (z * w) % n
    
    # Calculate u2 = r * w mod n
    u2 = (r * w) % n
    
    # Calculate point P = u1 * G + u2 * Q
    point = u1 * G + u2 * public_key
    
    # Check if point is at infinity
    if point == infinity_point:
        return False
    
    # Verify that r == point.x mod n
    is_valid_signature = (r == point.x() % n)
    
    return is_valid_signature
