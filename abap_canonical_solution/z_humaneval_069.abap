CLASS z_humaneval_069 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: search
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_069 IMPLEMENTATION.

  METHOD search.
    DATA: frq     TYPE TABLE OF i,
          max_val TYPE i,
          temp_i  TYPE i,
          ans     TYPE i VALUE -1.

    LOOP AT lst INTO temp_i.
      IF temp_i > max_val.
        max_val = temp_i.
      ENDIF.
    ENDLOOP.

    DO max_val + 1 TIMES.
      APPEND 0 TO frq.
    ENDDO.

    LOOP AT lst INTO temp_i.
      READ TABLE frq INDEX temp_i + 1 INTO DATA(current_freq).
      current_freq = current_freq + 1.
      MODIFY frq INDEX temp_i + 1 FROM current_freq.
    ENDLOOP.

    DO max_val TIMES.
      READ TABLE frq INDEX sy-index + 1 INTO current_freq.
      IF current_freq >= sy-index.
        ans = sy-index.
      ENDIF.
    ENDDO.

    result = ans.
  ENDMETHOD.

ENDCLASS.