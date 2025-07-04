import os
import json
import numpy as np
from openai import OpenAI
# Fix OPENAI_API_KEY passing from OS to Sandbox
import os
from dotenv import load_dotenv

class MetadataEmbedder:
    """Class for embedding metadata and storing them in a sandbox"""
    def __init__(self, sandbox=None):
        self.sandbox = sandbox
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Separate storage for metadata vs agent notes (using numpy instead of faiss)
        self.metadata_store_path = "embeddings/metadata_store.json"
        self.agent_notes_store_path = "embeddings/agent_notes_store.json"

        # Initialize stores (numpy-based)
        self.metadata_store = []
        self.agent_notes_store = []

    def _check_metadata_exists(self) -> bool:
        """Check if metadata embeddings already exist in sandbox"""
        if not self.sandbox:
            return False

        try:
            # Check if metadata store exists
            self.sandbox.files.read(self.metadata_store_path)
            return True
        except:
            return False

    def _load_existing_metadata(self):
        """Load existing metadata embeddings"""
        try:
            # Load metadata store (now includes embeddings)
            store_data = self.sandbox.files.read(self.metadata_store_path).decode()
            self.metadata_store = json.loads(store_data)

            print(f"Loaded existing metadata embeddings: {len(self.metadata_store)} items")
            return True
        except Exception as e:
            print(f"Error loading metadata embeddings: {e}")
            return False

    def embed_metadata_file(self, file_path: str, force_refresh: bool = False) -> str:
        """Embed the metadata markdown file at startup"""

        # Check if already exists and not forcing refresh
        if not force_refresh and self._check_metadata_exists():
            print("Metadata embeddings already exist, loading...")
            if self._load_existing_metadata():
                return "Metadata embeddings loaded successfully"

        print("Creating new metadata embeddings...")

        # Read the metadata file from sandbox
        try:
            metadata_content = self.sandbox.files.read(file_path)
            # Check if it's bytes or string
            if isinstance(metadata_content, bytes):
                metadata_content = metadata_content.decode()
        except Exception as e:
            return f"Error reading metadata file: {e}"

        # Split into chunks to make the file easier for the llm to read
        chunks = self._chunk_markdown(metadata_content)

        # Create embeddings
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                response = self.openai_client.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)

                # Store metadata about this chunk including embedding
                self.metadata_store.append({
                    "type": "metadata",
                    "source": "turtle_games_dataset_metadata.md",
                    "chunk_id": i,
                    "content": chunk,
                    "embedding": embedding,
                    "created_at": "startup"
                })
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
                continue

        if not embeddings:
            return "Error: No embeddings created"

        # Save to sandbox (embeddings are now stored in metadata_store)
        try:
            # Save metadata store with embeddings
            store_json = json.dumps(self.metadata_store, indent=2)
            self.sandbox.files.write(self.metadata_store_path, store_json.encode())

            return f"Successfully embedded metadata file: {len(chunks)} chunks created"

        except Exception as e:
            return f"Error saving metadata embeddings: {e}"

    def _chunk_markdown(self, content: str, chunk_size: int = 1000) -> list:
        """Split markdown content into chunks"""
        # Simple chunking by lines - you might want more sophisticated chunking
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            if current_size + len(line) > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def search_metadata(self, query: str, k: int = 3) -> list:
        """Search metadata embeddings using numpy cosine similarity"""
        if not self.metadata_store:
            return []

        try:
            # Create query embedding
            response = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = np.array(response.data[0].embedding)

            # Calculate similarities with all stored embeddings
            similarities = []
            for i, item in enumerate(self.metadata_store):
                if "embedding" in item:
                    stored_embedding = np.array(item["embedding"])
                    
                    # Normalize embeddings
                    norm_query = query_embedding / np.linalg.norm(query_embedding)
                    norm_stored = stored_embedding / np.linalg.norm(stored_embedding)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(norm_query, norm_stored)
                    similarities.append((similarity, i))

            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_similarities = similarities[:k]

            # Build results
            results = []
            for similarity, idx in top_similarities:
                result = self.metadata_store[idx].copy()
                result['similarity_score'] = float(similarity)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error searching metadata: {e}")
            return []

    def embed_tool_help_notes(self, tools: list) -> str:
        """
        Embeds the help_notes field from each tool into the metadata index.

        Args:
            tools (list): List of tool instances

        Returns:
            str: Success message with count of embedded help notes
        """
        if not tools:
            return "No tools provided"

        help_notes_count = 0
        embeddings = []

        for tool in tools:
            if hasattr(tool, "help_notes") and tool.help_notes:
                try:
                    # Create embedding for the help notes
                    response = self.openai_client.embeddings.create(
                        input=tool.help_notes,
                        model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)

                    # Store metadata about this tool help including embedding
                    self.metadata_store.append({
                        "type": "tool_help",
                        "tool_name": tool.name,
                        "content": tool.help_notes,
                        "embedding": embedding,
                        "created_at": "startup"
                    })
                    help_notes_count += 1

                except Exception as e:
                    print(f"Error embedding help notes for tool {tool.name}: {e}")
                    continue

        if embeddings:
            # Save updated store with embeddings
            try:
                store_json = json.dumps(self.metadata_store, indent=2)
                self.sandbox.files.write(self.metadata_store_path, store_json.encode())

            except Exception as e:
                return f"Error saving tool help embeddings: {e}"

        return f"Successfully embedded help notes for {help_notes_count} tools"