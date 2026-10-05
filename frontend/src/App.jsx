import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [verses, setVerses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askGita = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setVerses([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();

      setAnswer(data.answer);
      setVerses(data.verses || []);
    } catch (error) {
      console.error(error);

      setError(
        "Unable to connect to the Gita assistant. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askGita();
    }
  };

  return (
    <div className="min-h-screen bg-[#f7f3eb]">

      {/* Header */}
      <header className="border-b border-[#e5dccd] bg-[#fffaf2]">
        <div className="mx-auto max-w-5xl px-6 py-6">

          <div className="flex items-center gap-4">

            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[#8b5e34] text-2xl text-white">
              ॐ
            </div>

            <div>
              <h1 className="text-2xl font-bold text-[#3b2a1d]">
                Bhagavad Gita AI
              </h1>

              <p className="text-sm text-[#806f5c]">
                Guidance grounded in the teachings of the Gita
              </p>
            </div>

          </div>

        </div>
      </header>


      {/* Main */}
      <main className="mx-auto max-w-4xl px-6 py-12">

        {/* Hero */}
        <section className="mb-10 text-center">

          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-[#9a6b3f]">
            Ask • Reflect • Understand
          </p>

          <h2 className="text-4xl font-bold leading-tight text-[#30251d]">
            What is on your mind?
          </h2>

          <p className="mx-auto mt-4 max-w-2xl text-[#756657]">
            Describe a challenge, concern, or question and receive
            guidance grounded in relevant Bhagavad Gita verses.
          </p>

        </section>


        {/* Question Input */}
        <section className="rounded-3xl border border-[#e5dccd] bg-[#fffaf2] p-5 shadow-sm">

          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="For example: I am working hard but I am afraid I won't succeed..."
            rows={5}
            className="w-full resize-none rounded-2xl border border-[#ded2c2] bg-white p-4 text-[#30251d] outline-none transition focus:border-[#9a6b3f] focus:ring-2 focus:ring-[#9a6b3f]/20"
          />

          <div className="mt-4 flex items-center justify-between">

            <span className="text-xs text-[#968776]">
              Press Enter to ask
            </span>

            <button
              onClick={askGita}
              disabled={loading || !query.trim()}
              className="rounded-xl bg-[#8b5e34] px-6 py-3 font-semibold text-white transition hover:bg-[#704823] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Reflecting..." : "Get Guidance"}
            </button>

          </div>

        </section>


        {/* Error */}
        {error && (
          <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}


        {/* AI Answer */}
        {answer && (
          <section className="mt-10">

            <div className="mb-4 flex items-center gap-3">

              <div className="h-2 w-2 rounded-full bg-[#8b5e34]" />

              <h3 className="text-xl font-bold text-[#30251d]">
                Gita Guidance
              </h3>

            </div>

            <div className="rounded-3xl border border-[#e5dccd] bg-white p-7 shadow-sm">

              <div className="whitespace-pre-wrap leading-7 text-[#4d4035]">
                {answer}
              </div>

            </div>

          </section>
        )}


        {/* Retrieved Verses */}
        {verses.length > 0 && (
          <section className="mt-10">

            <h3 className="mb-5 text-xl font-bold text-[#30251d]">
              Retrieved Verses
            </h3>

            <div className="space-y-4">

              {verses.map((verse) => (
                <article
                  key={verse.verse_id}
                  className="rounded-2xl border border-[#e5dccd] bg-[#fffaf2] p-6"
                >

                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">

                    <span className="font-semibold text-[#8b5e34]">
                      Chapter {verse.chapter}, Verse {verse.verse}
                    </span>

                    <span className="rounded-full bg-[#eee4d5] px-3 py-1 text-xs text-[#756657]">
                      Similarity: {verse.similarity}
                    </span>

                  </div>

                  <p className="leading-7 text-[#4d4035]">
                    {verse.english}
                  </p>

                </article>
              ))}

            </div>

          </section>
        )}

      </main>


      {/* Footer */}
      <footer className="border-t border-[#e5dccd] py-8 text-center text-sm text-[#968776]">
        Bhagavad Gita RAG Assistant • MPNet + FAISS + Gemini
      </footer>

    </div>
  );
}

export default App;