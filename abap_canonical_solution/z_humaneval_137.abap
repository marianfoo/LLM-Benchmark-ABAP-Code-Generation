CLASS z_humaneval_137 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: compare_one
      IMPORTING
        a             TYPE any
        b             TYPE any
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_137 IMPLEMENTATION.

  METHOD compare_one.
    DATA: temp_a  TYPE string,
          temp_b  TYPE string,
          float_a TYPE f,
          float_b TYPE f.

    temp_a = a.
    temp_b = b.

    REPLACE ALL OCCURRENCES OF ',' IN temp_a WITH '.'.
    REPLACE ALL OCCURRENCES OF ',' IN temp_b WITH '.'.

    float_a = temp_a.
    float_b = temp_b.

    IF float_a = float_b.
      CLEAR result.
    ELSEIF float_a > float_b.
      result = a.
    ELSE.
      result = b.
    ENDIF.

    CONDENSE result.

  ENDMETHOD.

ENDCLASS.