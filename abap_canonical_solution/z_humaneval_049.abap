CLASS z_humaneval_049 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: modp
      IMPORTING
        n             TYPE i
        p             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_049 IMPLEMENTATION.

  METHOD modp.
    DATA: ret TYPE i VALUE 1,
          i   TYPE i.

    DO n TIMES.
      ret = ( 2 * ret ) MOD p.
    ENDDO.

    result = ret.
  ENDMETHOD.

ENDCLASS.