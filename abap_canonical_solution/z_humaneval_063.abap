CLASS z_humaneval_063 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: fibfib
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_063 IMPLEMENTATION.

  METHOD fibfib.
    IF n = 0.
      result = 0.
      RETURN.
    ENDIF.

    IF n = 1.
      result = 0.
      RETURN.
    ENDIF.

    IF n = 2.
      result = 1.
      RETURN.
    ENDIF.

    result = fibfib( n - 1 ) + fibfib( n - 2 ) + fibfib( n - 3 ).
  ENDMETHOD.

ENDCLASS.