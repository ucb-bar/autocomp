import { useState, useRef, useEffect } from "react";

interface Option {
  id: string;
  label: string;
}

interface DropdownProps {
  options: Option[];
  value: string;
  onChange: (id: string) => void;
}

export default function Dropdown({ options, value, onChange }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const selected = options.find((o) => o.id === value) ?? options[0];

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs font-mono bg-white border border-stone-200 rounded-md pl-2.5 pr-7 py-1.5 text-stone-700 shadow-sm hover:border-stone-300 focus:outline-none focus:ring-1 focus:ring-indigo-300 cursor-pointer text-left min-w-[10rem] max-w-xs truncate"
      >
        <span className="truncate">{selected?.label ?? "—"}</span>
        <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-stone-400 text-[10px]">▼</span>
      </button>
      {open && (
        <div className="absolute z-50 mt-1 left-0 w-max min-w-full max-w-sm bg-white border border-stone-200 rounded-md shadow-lg py-1 max-h-52 overflow-y-auto">
          {options.map((o) => (
            <button
              key={o.id}
              onClick={() => { onChange(o.id); setOpen(false); }}
              className={`block w-full text-left px-3 py-1.5 text-xs font-mono truncate transition-colors ${
                o.id === value
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-stone-600 hover:bg-stone-50"
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
