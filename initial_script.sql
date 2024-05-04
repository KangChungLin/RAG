CREATE EXTENSION vector;

CREATE TABLE public.langchain_pg_collection (
	"name" varchar NULL,
	uuid uuid NOT NULL,
	role_name varchar NULL,
	job varchar NULL,
	update_time timestamp NULL DEFAULT now(),
	CONSTRAINT langchain_pg_collection_pkey PRIMARY KEY (uuid)
);

CREATE TABLE public.langchain_pg_document (
	uid varchar NOT NULL,
	collection_id uuid NULL,
	"document" varchar NULL,
	title varchar NOT NULL,
	category varchar NULL,
	url varchar NULL,
	publish_time timestamp NULL,
	update_time timestamp NULL DEFAULT now(),
	CONSTRAINT langchain_pg_document_pkey PRIMARY KEY (uid),
	CONSTRAINT langchain_pg_document_collection_id_fkey FOREIGN KEY (collection_id) REFERENCES public.langchain_pg_collection(uuid) ON DELETE CASCADE
);

CREATE TABLE public.langchain_pg_embedding (
	uuid uuid NOT NULL,
	document_id varchar NULL,
	embedding public.vector NULL,
	paragraph varchar NULL,
	CONSTRAINT langchain_pg_embedding_pkey PRIMARY KEY (uuid),
	CONSTRAINT langchain_pg_embedding_collection_id_fkey FOREIGN KEY (document_id) REFERENCES public.langchain_pg_document(uid) ON DELETE CASCADE
);