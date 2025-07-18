CLASS z_humaneval_097 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: multiply
      IMPORTING
        a             TYPE i
        b             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_097 IMPLEMENTATION.

  METHOD multiply.
    result = abs( a MOD 10 ) * abs( b MOD 10 ).
  ENDMETHOD.

ENDCLASS.