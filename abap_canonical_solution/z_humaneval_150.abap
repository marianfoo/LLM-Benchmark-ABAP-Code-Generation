CLASS z_humaneval_150 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: x_or_y
      IMPORTING
        VALUE(n)      TYPE i
        VALUE(x)      TYPE i
        VALUE(y)      TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_150 IMPLEMENTATION.

  METHOD x_or_y.
    DATA: i TYPE i.

    IF n = 1.
      result = y.
      RETURN.
    ENDIF.

    DO n - 2 TIMES.
      i = sy-index + 1.
      IF n MOD i = 0.
        result = y.
        RETURN.
      ENDIF.
    ENDDO.

    result = x.
  ENDMETHOD.

ENDCLASS.