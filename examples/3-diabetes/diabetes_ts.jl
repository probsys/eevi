using Gen:
    @gen,
    @trace,
    categorical

include("diabetes_cpds.jl")

@gen function model(n::Integer)
    meal::Vector{Int} = Vector{Int}(undef, n)
    cho::Vector{Int} = Vector{Int}(undef, n)
    bg::Vector{Int} = Vector{Int}(undef, n)
    activ_ins::Vector{Int} = Vector{Int}(undef, n)
    ins_abs::Vector{Int} = Vector{Int}(undef, n)
    gut_abs::Vector{Int} = Vector{Int}(undef, n)
    renal_cl::Vector{Int} = Vector{Int}(undef, n)
    ins_indep_util::Vector{Int} = Vector{Int}(undef, n)
    ins_dep_util::Vector{Int} = Vector{Int}(undef, n)
    glu_prod::Vector{Int} = Vector{Int}(undef, n)
    ins_indep::Vector{Int} = Vector{Int}(undef, n)
    ins_dep::Vector{Int} = Vector{Int}(undef, n)
    endo_bal::Vector{Int} = Vector{Int}(undef, n)
    cho_bal::Vector{Int} = Vector{Int}(undef, n)
    basal_bal::Vector{Int} = Vector{Int}(undef, n)
    met_irr::Vector{Int} = Vector{Int}(undef, n)
    tot_bal::Vector{Int} = Vector{Int}(undef, n)

    # global: cho_init
    p_cho_init = [.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .85]
    cho_init = @trace(categorical(p_cho_init), :cho_init)

    # global: ins_sens
    p_ins_sens = [.2, .2, .2, .2, .2]
    ins_sens = @trace(categorical(p_ins_sens), :ins_sens)

    for i=1:n

        # meal
        p_meal = [.01, .01, .01, .01, .01, .01, .01, .01, .01, .01,.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .8]
        meal[i] = @trace(categorical(p_meal), :meal => i)

        # cho
        cho_parent = i == 1 ? cho_init : cho_bal[i-1]
        p_cho = CPT_cho[(meal[i], cho_parent)]
        cho[i] = @trace(categorical(p_cho), :cho => i)

        # gut_abs
        p_gut_abs = CPT_gut_abs[cho[i]]
        gut_abs[i] = @trace(categorical(p_gut_abs), :gut_abs => i)

        # cho_bal
        p_cho_bal = CPT_cho_bal[(cho[i], gut_abs[i])]
        cho_bal[i] = @trace(categorical(p_cho_bal), :cho_bal => i)

        # bg
        if i == 1
            p_bg = [.01, .01, .02, .07, .15, .3, .2, .15, .07, .01, .01]
        else
            p_bg = CPT_bg[(bg[i-1], tot_bal[i-1])]
        end
        bg[i] = @trace(categorical(p_bg), :bg => i)

        # ins_abs
        p_ins_abs = [.03, .04, .04, .04, .1, .15, .2, .2, .1, .06, .04]
        ins_abs[i] = @trace(categorical(p_ins_abs), :ins_abs => i)

        # activ_ins
        p_activ_ins = CPT_activ_ins[(ins_abs[i], ins_sens)]
        activ_ins[i] = @trace(categorical(p_activ_ins), :active_ins => i)

        # renal_cl
        p_renal_cl = CPT_renal_cl[bg[i]]
        renal_cl[i] = @trace(categorical(p_renal_cl), :renal_cl => i)

        # ins_indep_util
        p_ins_indep_util = CPT_ins_indep_util[bg[i]]
        ins_indep_util[i] = @trace(categorical(p_ins_indep_util), :ins_indep_util => i)

        # ins_dep_util
        p_ins_dep_util = CPT_ins_dep_util[(bg[i], activ_ins[i])]
        ins_dep_util[i] = @trace(categorical(p_ins_dep_util), :ins_dep_util => i)

        # glu_prod
        p_glu_prod = CPT_glu_prod[(activ_ins[i], bg[i])]
        glu_prod[i] = @trace(categorical(p_glu_prod), :glu_prod => i)

        # ins_indep
        p_ins_indep = CPT_ins_indep[(renal_cl[i], ins_indep_util[i])]
        ins_indep[i] = @trace(categorical(p_ins_indep), :ins_indep => i)

        # ins_dep
        p_ins_dep = CPT_ins_dep[(ins_dep_util[i], glu_prod[i])]
        ins_dep[i] = @trace(categorical(p_ins_dep), :ins_dep => i)

        # endo_bal
        p_endo_bal = CPT_endo_bal[(ins_indep[i], ins_dep[i])]
        endo_bal[i] = @trace(categorical(p_endo_bal), :endo_bal => i)

        # basal_bal
        p_basal_bal = CPT_basal_bal[(gut_abs[i], endo_bal[i])]
        basal_bal[i] = @trace(categorical(p_basal_bal), :basal_bal => i)

        # met_irr
        p_met_irr = [.02, .05, .18, .5, .18, .05, .02]
        met_irr[i] = @trace(categorical(p_met_irr), :met_irr => i)

        # tot_bal
        p_tot_bal = CPT_tot_bal[(basal_bal[i], met_irr[i])]
        tot_bal[i] = @trace(categorical(p_tot_bal), :tot_bal => i)
    end
end
