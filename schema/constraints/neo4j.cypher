// Node constraints for unique identifiers
CREATE CONSTRAINT gene_pk IF NOT EXISTS
FOR (g:Gene) REQUIRE g.hgnc_id IS UNIQUE;

CREATE CONSTRAINT gene_symbol IF NOT EXISTS
FOR (g:Gene) REQUIRE g.symbol IS UNIQUE;

CREATE CONSTRAINT chemical_pk IF NOT EXISTS
FOR (c:Chemical) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT chemical_rxnorm IF NOT EXISTS
FOR (c:Chemical) REQUIRE c.rxnorm_id IS UNIQUE;

CREATE CONSTRAINT mutation_pk IF NOT EXISTS
FOR (m:Mutation) REQUIRE m.hgvs IS UNIQUE;

CREATE CONSTRAINT disease_pk IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT disease_ncit IF NOT EXISTS
FOR (d:Disease) REQUIRE d.ncit_id IS UNIQUE;

CREATE CONSTRAINT phenotype_pk IF NOT EXISTS
FOR (p:Phenotype) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT pathway_pk IF NOT EXISTS
FOR (pw:Pathway) REQUIRE pw.name IS UNIQUE;

CREATE CONSTRAINT model_pk IF NOT EXISTS
FOR (m:Model) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT finding_pk IF NOT EXISTS
FOR (f:Finding) REQUIRE f.description IS UNIQUE;

// Index for Paper/Publication nodes
CREATE INDEX paper_pmid IF NOT EXISTS
FOR (p:Paper) ON (p.pmid);

// Indexes for faster lookups
CREATE INDEX gene_symbol_idx IF NOT EXISTS
FOR (g:Gene) ON (g.symbol);

CREATE INDEX gene_entrez_idx IF NOT EXISTS
FOR (g:Gene) ON (g.entrez_id);

CREATE INDEX chemical_name_idx IF NOT EXISTS
FOR (c:Chemical) ON (c.name);

CREATE INDEX chemical_chebi_idx IF NOT EXISTS
FOR (c:Chemical) ON (c.chebi_id);

CREATE INDEX mutation_rsid_idx IF NOT EXISTS
FOR (m:Mutation) ON (m.rsid);

CREATE INDEX mutation_gene_idx IF NOT EXISTS
FOR (m:Mutation) ON (m.gene_symbol);

CREATE INDEX disease_doid_idx IF NOT EXISTS
FOR (d:Disease) ON (d.doid);

CREATE INDEX pathway_reactome_idx IF NOT EXISTS
FOR (pw:Pathway) ON (pw.reactome_id);

CREATE INDEX model_type_idx IF NOT EXISTS
FOR (m:Model) ON (m.type);
