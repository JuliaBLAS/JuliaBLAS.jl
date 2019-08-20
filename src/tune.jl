
function get_hw_params()
    topology = Hwloc.topology_load()
    summary = Hwloc.getinfo(topology)
    l3 = findcache(topology, :L3Cache).attr
    # assume L2 caches are of same attributes
    l2 = findcache(topology, :L2Cache).attr
    l2ct = countcache(topology, :L2Cache)
    # assume L1 caches are of same attributes
    l1 = findcache(topology, :L1Cache).attr
    l1ct = countcache(topology, :L2Cache)
end

function findcache(v, sym)
    for c in v.children
        if c.type_ == sym
            return c
        else
            return findcache(c, sym)
        end
    end
end

function countcache(v, sym)
    ct = 0
    for c in v.children
        c.type_ == sym && (ct += 1)
        ct += countcache(c,sym)
    end
    return ct
end