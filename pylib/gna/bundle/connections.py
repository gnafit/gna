from __future__ import print_function
debug_connections=True

def each_pair( sequence ):
    """Take each pair from a sequence (overlapping):
        each_pair( [1,2,3, 4] ) will yield (1,2), (2,3) and (3,4)"""
    if len(sequence)<2:
        raise Exception( 'Should define at least two elements for iteration' )
    it = iter( sequence )
    prev = it.next()
    try:
        while True:
            current = it.next()
            yield prev, current
            prev = current
    except StopIteration:
        pass

def transformations_map( chains, connections ):
    """ Connects sequences of transformations input-to-output
    chains = (
        (class1_instance1, class1_instance_2,..., class1_instance_N),
        (class2_instance1, class2_instance_2,..., class2_instance_N),
        (class3_instance1, class3_instance_2,..., class3_instance_N),
    )
    will connect each column, i.e. same index instances of various classes:
        class1_instance1->class2_instance1->class3_instance1,
        class1_instance2->class2_instance2->class3_instance2,
        ...
        class1_instanceN->class2_instanceN->class3_instanceN,

    inputs and outputs are specified by
    connections = [
        [ ( class1_transformation, output1 ), ( class2_transformation, input2 ) ],
        [ ( class2_transformation, output2 ), ( class3_transformation, input3 ) ],
        ...
        [ ( classN-1_transformation, outputN-1 ), ( classN_transformation, inputN ) ],
    ]
    connections is a list of pairs(source, target) where source/target is a pair(transformation, input/output)

    """
    for i, chain in enumerate(zip(*chains)):
        if debug_connections:
            print( 'Chain', i )

        for j, ((source, target), ((source_trans, output), (target_trans, input))) in enumerate(zip(each_pair(chain), connections)):
            if debug_connections:
                print( '  connection {idx:02d} {sclass}.{strans}.{soutput} -> {tclass}.{ttrans}.{tinput}'.format(
                    idx=j,
                    sclass=type(source).__name__, strans=source_trans, soutput=output,
                    tclass=type(target).__name__, ttrans=target_trans, tinput=input
                    ) )
            target.transformations[target_trans].inputs[input]( source.transformations[source_trans].outputs[output] )
