CLASS z_humaneval_055 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: fib
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_055 IMPLEMENTATION.

  METHOD fib.
    IF n = 0.
      result = 0.
      RETURN.
    ENDIF.

    IF n = 1.
      result = 1.
      RETURN.
    ENDIF.

    result = fib( n - 1 ) + fib( n - 2 ).
  ENDMETHOD.

ENDCLASS.