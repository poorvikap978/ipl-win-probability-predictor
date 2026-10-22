
/* Experimental ML probability display */
.probability - grid {
    display: grid;
    grid - template - columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin: 16px 0;
}

.probability - card {
    min - width: 0;
    padding: 16px;
    border: 1px solid #d9dee8;
    border - radius: 12px;
    background: #ffffff;
    color: #202938;
}

.probability - card h4 {
    margin: 0 0 10px;
    overflow - wrap: anywhere;
}

.probability - value {
    margin: 0 0 12px;
    font - size: 1.6rem;
    font - weight: 700;
}

.probability - track {
    height: 10px;
    overflow: hidden;
    border - radius: 999px;
    background: #e6eaf0;
}

.probability - fill {
    height: 100 %;
    border - radius: inherit;
    background: #2563eb;
    transition: width 0.3s ease;
}

.probability - disclaimer {
    padding: 12px;
    border - radius: 8px;
    background: #fff7e6;
    color: #704b00;
    font - size: 0.9rem;
    line - height: 1.5;
}

@media(max - width: 480px) {
    .probability - grid {
        grid - template - columns: 1fr;
    }
}
