import { useEffect, useId, useRef, useState } from "react";

interface InfoHelpProps {
  label: string;
  text: string;
}

export function InfoHelp({ label, text }: InfoHelpProps) {
  const [pinnedOpen, setPinnedOpen] = useState(false);
  const [hovered, setHovered] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const tooltipId = useId();
  const isOpen = pinnedOpen || hovered;

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!wrapperRef.current?.contains(event.target as Node)) {
        setPinnedOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, []);

  return (
    <div
      ref={wrapperRef}
      className={`info-help ${isOpen ? "open" : ""}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        className="info-help__button"
        aria-label={label}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? tooltipId : undefined}
        onClick={() => setPinnedOpen((current) => !current)}
        onFocus={() => setHovered(true)}
        onBlur={() => setHovered(false)}
      >
        ?
      </button>
      {isOpen ? (
        <div id={tooltipId} role="tooltip" className="info-help__popover">
          <p>{text}</p>
        </div>
      ) : null}
    </div>
  );
}
