import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from ovito.io import import_file
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier
)
from ovito.data import NearestNeighborFinder

        
### Single Atom analysis ###
def neighbor_analysis(path, SASA_outfile, gro_file):
    # Load positions from the spec file
    pos=np.loadtxt(Path(path) / SASA_outfile, skiprows=2, usecols=(1,2,3))
    # create dict to store the nighbour informations
    neighbor ={'ID':[], 'ParticleType': [], 'ResidueType':[], 'AtomName':[]}
    # Load the gro file to find the nearest nighbours of the SASA atoms
    pipeline = import_file(Path(path) / gro_file)
    # Delete the solvent and Ions
    with open(gro_file, "r") as rf:
        for i,line in enumerate(rf):
            if 'SOL' in line or 'SOD' in line:
                atmnr = i - 2
                break
    # UPO_del_Solv - Expression selection:
    pipeline.modifiers.append(ExpressionSelectionModifier(
        expression = 'ParticleIdentifier>%s'%str(atmnr)))
    #  UPO_del_Solv - Delete selected
    pipeline.modifiers.append(DeleteSelectedModifier())
    d = pipeline.compute()
    # Initialize neighbor finder object and visit the 1 nearest neighbors of each particle.
    finder = NearestNeighborFinder(1, d)
    # Visit particles closest to some spatial point (x,y,z):
    for i in range(len(pos)):
        for neigh in finder.find_at(pos[i]):
            #print(neigh.index)
            neighbor['ID'].append(d.particles['Particle Identifier'][neigh.index])
            neighbor['ParticleType'].append(d.particles['Particle Type'] \
                    .type_by_id(d.particles['Particle Type'][neigh.index]).name)
            neighbor['ResidueType'].append(d.particles['Residue Type'] \
                    .type_by_id(d.particles['Residue Type'][neigh.index]).name)
            neighbor['AtomName'].append(d.particles['Atom Name'] \
                    .type_by_id(d.particles['Atom Name'][neigh.index]).name)
            #neighbor['dist'].append(neigh.distance)
    neighbor = pd.DataFrame(neighbor)
    return neighbor

def atom_analysis(path, SASA_outfile, neighbor):
    # load SASA results
    spec = pd.read_csv(Path(path) / SASA_outfile, sep='\s+', skiprows=1).drop(['atom'], axis=1)
    # insert neighbor properties into df with energy results, delete dublicates and sort by lowest energy
    for key in neighbor:
        spec.insert(0, key, neighbor[key])
        # sort all values by lowest energy and define cutoff for strongest interactions (30 values)
    spec=spec.sort_values('eint/eV', ascending=True).drop_duplicates(subset=['ID']).reset_index(drop=True)
    strongest_interaction = spec.iloc[:30]
    # save strongest interactions to file
    strongest_interaction.to_csv('./atom_analysis_total.txt', sep='\t', index=False)
    # count the amount of most attacked particle types and devide by their total number of all attacked (not in general in the protein!)
    total_count = spec['ParticleType'].value_counts()
    s_count = strongest_interaction['ParticleType'].value_counts()
    result = {'labels': [],'total':[], 'percent': [],}
    for item in s_count.index:
        result['labels'].append(str(item))
        result['total'].append(s_count[item])
        result['percent'].append(float(s_count[item]/total_count[item]))
    pd.DataFrame(result).to_csv(Path(path) / 'atom_analysis_percent.txt', 
                                sep='\t', index=False)
    return result

def atom_analysis_plot(path, neighbor, result):
    # set up color list
    colors={'C': 'grey', 'H': 'whitesmoke', 'N': 'b', 'O': 'r', 'S':'y', 'Fe':'organge', 'Mg': 'm'}
    # set up plot
    fig, ax = plt.subplots(1)
    fig.set_size_inches(4, 5)
    ## plot bars ##
    ax.bar(result['labels'],result['total'],edgecolor='black',
            color=[colors[key] for key in result['labels']],)
    #layout
    ax.set_ylabel('Total interaction',fontsize=18)
    ax.set_xlabel('Particle type',fontsize=18)
    ## increase line thickness box and ticks      
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(labelsize=16, width=1.5, length=6, bottom=True, left=True, right=True,)
    ## minor ticks
    #ax.minorticks_on()
    ax.tick_params(which='minor',width=1, length=6, right=True,)
    #save
    plt.savefig(Path(path) / 'attack_vs_atomtype.png', dpi = 300, bbox_inches='tight')
    plt.show()

### Residue specific analysis ###
def residuelist(path, gro_file):
    # read .gro file
    gro = pd.read_csv(Path(path) / gro_file, sep='\s+', skiprows=2, 
                names=['ResType', 'AtomType', 'AtomNumber', 'x', 'y', 'z', 'vx', 'vy', 'vz',] )
    # find the place to cut the dataset (when the waters or ions start) and ignore SOL and counter Ions
    for i,item in enumerate(gro['ResType']):
        if 'SOL' in item or 'SOD' in item:
            index = i
            break
        else:
            pass
    gro = gro[gro.index < index]
    residuelist = pd.DataFrame([re.split(r'(\d+)', s) for s in gro['ResType']]).drop(columns = 0).drop_duplicates(subset=1)
    np.savetxt(Path(path) / 'residuelist.txt', residuelist,fmt='%s', delimiter='  ', 
                header='ResNum ResType', comments='')
    return residuelist

def residue_analysis(path, SASA_outfile, residuelist):
    df = pd.read_csv(Path(path) / SASA_outfile, sep='\s+', skiprows=1)
    res = pd.read_csv(Path(path) / residuelist, sep='\s+',)
    # sort by lowest energy and identify and remove dublicates of the same residue
    sort= df.sort_values('eint/eV', ascending=True).reset_index(drop=True)
    # only take the 30 lowest
    emin=sort.drop_duplicates(subset=['res']).iloc[0:30]
    for item in emin['res']:
        loc= res.isin([item]).any(axis=1).idxmax()
        resn = res.iloc[loc]['ResType']
        emin = emin.replace(item, resn)
    emin.to_csv(Path(path) / 'minimum_energy_values.txt', sep='\t', index=False)
    # get all residues and count, count residues in emin
    res_names=res['ResType'].reset_index(drop=True)
    res_names = res_names.value_counts()
    count = emin['res'].value_counts()
    # get normalized results, attack amount/total amount of residue type in the protein
    result = {'labels': [], 'total':[], 'enrichment': [],}
    for item in count.index:
        result['total'].append(count[item])
        result['labels'].append(str(item))
        # Calculate relative values/enrichment as explained in the publication
        expectation_value = (res_names[item]/res.count()[0])*30
        enrichment = count[item] / expectation_value
        result['enrichment'].append(float(enrichment))
    # export attack amount vs residue to file
    pd.DataFrame(result).to_csv(Path(path) / 'interaction_probability.txt', sep='\t', 
            index=False)
    return result

def residue_analysis_plot(path, result):
    # LOAD AND PREPARE
    result = pd.DataFrame(result).sort_values('labels', ascending=True)
    ## Create color list with fixed color for each residue
    colors={'ALA': '#d6a090',
        'ARG': '#fe3b1e',
        'ASN': '#a12c32',
        'ASP': '#fa2f7a',
        'CYM': '#fb9fda',
        'CYS': '#e61cf7',
        'GLN': '#992f7c',
        'GLU': '#11963b',
        'GLY': '#051155',
        'HEM': '#4f02ec',
        'HIS': '#2d69cb',
        'ILE': '#08a29a',
        'LEU': '#2a666a',
        'LYS': '#063619',
        'MET': '#000000',
        'MG': '#4a4957',
        'PHE': '#8e7ba4',
        'PRO': '#b7c0ff',
        'SER': '#ffffff',
        'THR': '#acbe9c',
        'TRP': '#827c70',
        'TYR': '#5a3b1c',
        'VAL': '#ae6507'}
    labels={'ALA': 'Ala',
        'ARG': 'Arg',
        'ASN': 'Asn',
        'ASP': 'Asp',
        'CYM': 'Cym',
        'CYS': 'Cys',
        'GLN': 'Gln',
        'GLU': 'Glu',
        'GLY': 'Gly',
        'HEM': 'Heme',
        'HIS': 'His',
        'ILE': 'Ile',
        'LEU': 'Leu',
        'LYS': 'Lys',
        'MET': 'Met',
        'MG': 'Mg',
        'PHE': 'Phe',
        'PRO': 'Pro',
        'SER': 'Ser',
        'THR': 'Thr',
        'TRP': 'Trp',
        'TYR': 'Tyr',
        'VAL': 'Val'}
    ## Add all res names to results and add zeros 
    result_keys = set(result["labels"])
    labels_keys = set(labels.keys())
    diff = labels_keys.difference(result_keys)
    df = pd.DataFrame({key: [0.0] for key in diff}).T.reset_index()
    r1 = result['labels'].append(df['index']).reset_index(drop=True)
    r2 = result['total'].append(df[0]).reset_index(drop=True)
    r3 = result['enrichment'].append(df[0]).reset_index(drop=True)
    r = pd.concat([r1,r2], axis = 1,).rename(columns={0 : 'labels', 1 : 'total'}).sort_values('labels', ascending=True)
    s = pd.concat([r1,r3], axis = 1,).rename(columns={0 : 'labels', 1 : 'enrichment'}).sort_values('labels', ascending=True)
    # PLOT: attack amount vs residue
    fig, ax = plt.subplots(2, 1, sharex=True)
    ## Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(10,7)
    fig.supylabel('Interactions with rmdPGS',fontsize=24, x=-0.05, fontweight='bold')
    ## Plot bars
    ax[1].bar(r['labels'],r['total'],edgecolor='black', 
            color=[colors[key] for key in r['labels']], label = '$n$ total', width = 0.6)
    # LAYOUT
    ax[1].margins(x=0.01)
    ax[1].set_ylabel('$n$ total',fontsize=24, labelpad=10)
    ax[1].set_xlabel('residue',fontsize=24, )
    ## Increase line thickness box and ticks      
    for axis in ['top','bottom','left','right']:
        ax[1].spines[axis].set_linewidth(1.5)
    ax[1].tick_params(labelsize=24, width=1.5, length=8, bottom=True, left=True, right=True,)
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    ax[1].set_xticklabels([labels[key] for key in r['labels']], fontsize=22)
    ## Minor ticks
    ax[1].minorticks_on()
    ax[1].tick_params(which='minor',width=1, length=6, right=True, bottom=False)
    ax[1].set_ylim(0,7)
    ## Plot bars 2
    ax[0].bar(s['labels'],s['enrichment'],edgecolor='black', 
            color=[colors[key] for key in r['labels']], label = 'relative / %', width = 0.6)
    ax[0].margins(x=0.01)
    ax[0].set_ylabel('relative',fontsize=24, labelpad=40)
    ax[0].set_xlabel('residue',fontsize=24)
    ## Increase line thickness box and ticks      
    for axis in ['top','bottom','left','right']:
        ax[0].spines[axis].set_linewidth(1.5)
    ax[0].tick_params(labelsize=24, width=1.5, length=8, bottom=True, left=True, right=True,)
    ax[0].tick_params(which='minor',width=1, length=6, right=True,)
    ax[0].minorticks_on()
    # SAVE
    plt.savefig(Path(path) / 'attack_vs_residue_enrichment.png', dpi = 300, bbox_inches='tight')
    plt.show()
