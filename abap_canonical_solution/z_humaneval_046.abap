CLASS z_humaneval_046 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: fib4
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.
ENDCLASS.

CLASS z_humaneval_046 IMPLEMENTATION.
  METHOD fib4.
    DATA: results TYPE int4_table,
          sum     TYPE i,
          i       TYPE i.

    results = VALUE #( ( 0 ) ( 0 ) ( 2 ) ( 0 ) ).

    IF n < 4.
      READ TABLE results INDEX ( n + 1 ) INTO result.
      RETURN.
    ENDIF.

    DO ( n - 3 ) TIMES.
      sum = results[ 1 ] + results[ 2 ] + results[ 3 ] + results[ 4 ].
      APPEND sum TO results.
      DELETE results INDEX 1.
    ENDDO.

    result = results[ 4 ].
  ENDMETHOD.
ENDCLASS.