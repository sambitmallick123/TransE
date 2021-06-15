# TransE
TransE is an energy-based model that produces knowledge base embeddings. It models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities. Relationships are represented as translations in the embedding space: if (h,l,t) holds, the embedding of the tail entity t should be close to the embedding of the head entity h plus some vector that depends on the relationship l.


Ref :
Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. and Yakhnenko, O., 2013, December. Translating embeddings for modeling multi-relational data. In Neural Information Processing Systems (NIPS) (pp. 1-9).
