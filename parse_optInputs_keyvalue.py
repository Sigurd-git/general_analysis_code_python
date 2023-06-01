def parse_optInputs_keyvalue(optargs, I=None, *varargin):
    """
    Parse optional inputs specified as a key-value pair. Key must be a string.

    I is an optional input argument with a structure containing the default values
    of the parameters, which are overwritten by the optional arguments. If I is
    specified then every key must match one of the fields of the structure I.

    C is a dictionary that tells you whether each field has been modified.

    C_value has the value for all of the fields that have been changed

    -- Example --

    # parse key-value pairs
    optargs = ['key1', [1, 2, 3], 'key2', list(range(1, 4)), 'key3', 'abc']
    I, C = parse_optInputs_keyvalue(optargs)

    # specifying default values
    I = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C = parse_optInputs_keyvalue(['key1', [4, 5, 6]], I)

    # empty lists mean unspecified
    I = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C = parse_optInputs_keyvalue(['key1', []], I, 'empty_means_unspecified', True)

    # use defaults to catch a spelling mistake
    I = parse_optInputs_keyvalue(['keys1', [4, 5, 6]], I)

    """

    P = {
        'empty_means_unspecified': False,
        'ignore_bad_keys': False,
        'ignore_mismatch_class': [],
        'always_include': [],
        'always_exclude': [],
        'maxlen': 100,
        'delimiter': '/',
        'noloop': False,
        'paramstring': False
    }

    if varargin:
        if varargin[0] == 'noloop':
            P['noloop'] = True
        else:
            P,_,_,_,_ = parse_optInputs_keyvalue(varargin, P, 'noloop')
            if P['empty_means_unspecified'] is None:
                P['empty_means_unspecified'] = False

    n_optargs = len(optargs)
    if n_optargs % 2 != 0:
        raise ValueError('There are not an even number of optional inputs')

    if I is None:
        I = {}
    else:
        possible_keys = list(I.keys())

    if len(set(optargs[0:n_optargs:2])) != len(optargs[0:n_optargs:2]):
        raise ValueError('Duplicate keys')

    if 'possible_keys' in locals():
        C = {key: False for key in possible_keys}
    else:
        C = {}

    C_value = {}

    if n_optargs == 0 and not P['paramstring']:
        all_keys = []
        return I, C, C_value, all_keys, ''

    i_key = list(range(0, n_optargs, 2))
    i_val = list(range(1, n_optargs, 2))
    n_pairs = n_optargs // 2
    all_keys = [optargs[i] for i in i_key]
    for j in range(n_pairs):
        key = optargs[i_key[j]]
        value = optargs[i_val[j]]

        if not isinstance(key, str):
            raise ValueError('Optional arguments not formatted properly\nAll keys must be strings\n')

        if 'possible_keys' in locals():
            if key not in possible_keys:
                if P['ignore_bad_keys']:
                    continue
                else:
                    raise ValueError("Optional arguments not formatted properly\n'{}' not a valid key\n".format(key))

            if type(I[key]) != type(value) and key not in P['ignore_mismatch_class']:
                allowed_class_swaps = [('double', 'int32')]
                allowed_swap = False
                for class_swap in allowed_class_swaps:
                    if P['empty_means_unspecified'] and value == []:
                        allowed_swap = True
                    if type(I[key]).__name__ == class_swap[0] and type(value).__name__ == class_swap[1]:
                        allowed_swap = True
                    if type(value).__name__ == class_swap[0] and type(I[key]).__name__ == class_swap[1]:
                        allowed_swap = True
                if not allowed_swap:
                    raise ValueError("Optional arguments not formatted properly\nValue of '{}' should be of type {}\n".format(key, type(I[key]).__name__))

        if P['empty_means_unspecified'] and value == []:
            continue
        else:
            if key not in I or I[key] != value:
                I[key] = value
                C[key] = True
                C_value[key] = I[key]

    C_value = {key: C_value[key] for key in set(I.keys()).intersection(set(C_value.keys()))}
    C_value = dict(sorted(C_value.items(), key=lambda x: list(I.keys()).index(x[0])))

    if P['paramstring'] and not P['noloop']:
        paramstring = optInputs_to_string(I, C_value, P['always_include'], P['always_exclude'], maxlen=P['maxlen'], delimiter=P['delimiter'])
    else:
        paramstring = ''

    return I, C, C_value, all_keys, paramstring


if __name__ == '__main__':

    # parse key-value pairs
    optargs = ['key1', [1, 2, 3], 'key2', list(range(1, 4)), 'key3', 'abc']
    I, C,_,_,_ = parse_optInputs_keyvalue(optargs)

    # specifying default values
    I = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C,_,_,_ = parse_optInputs_keyvalue(['key1', [4, 5, 6]], I)

    # empty lists mean unspecified
    I = {'key1': [1, 2, 3], 'key2': list(range(1, 4)), 'key3': 'abc'}
    I, C,_,_,_ = parse_optInputs_keyvalue(['key1', []], I, 'empty_means_unspecified', True)

    # use defaults to catch a spelling mistake
    I,_,_,_,_ = parse_optInputs_keyvalue(['keys1', [4, 5, 6]], I)
