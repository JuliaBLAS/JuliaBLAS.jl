
struct BlockAttributes
    mc::Int
    kc::Int
    nc::Int
    mr::Int
    nr::Int
end

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

    #defaults
        mc = 72
        kc = 192
        nc = 4080

    wl1 = div((l1.size*l1ct),l1.linesize)
    # add + mul latency or FMA latency
    lvfma = 5+3 # TODO: auto?
    # 
    nvec = 4
    sdata = 64
    nvfma = 1

    mr = ceil(Int, sqrt(nvec*lvfma*nvfma)/nvec)*nvec
    nr = ceil(Int, nvec*lvfma*nvfma)

    car = floor(Int, (wl1 - 1)/(1+nr/mr))
    cbr = ceil(Int, nr*car/mr)

    #kc = car*l1.size*l1.linesize/(mr*sdata)

    BlockAttributes(mc, kc, nc, mr, nr)
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