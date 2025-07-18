CLASS ltc_test_compare DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
    METHODS test_case_4 FOR TESTING.

ENDCLASS.

CLASS ltc_test_compare IMPLEMENTATION.

  METHOD test_case_1.
    DATA: game     TYPE TABLE OF i,
          guess    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    game = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 1 ) ).
    guess = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 2 ) ( -2 ) ).
    expected = VALUE #( ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 3 ) ( 3 ) ).

    result = z_humaneval_152=>compare( game = game guess = guess ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: game     TYPE TABLE OF i,
          guess    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    game = VALUE #( ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ).
    guess = VALUE #( ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ).
    expected = VALUE #( ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ( 0 ) ).

    result = z_humaneval_152=>compare( game = game guess = guess ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: game     TYPE TABLE OF i,
          guess    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    game = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    guess = VALUE #( ( -1 ) ( -2 ) ( -3 ) ).
    expected = VALUE #( ( 2 ) ( 4 ) ( 6 ) ).

    result = z_humaneval_152=>compare( game = game guess = guess ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: game     TYPE TABLE OF i,
          guess    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    game = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 5 ) ).
    guess = VALUE #( ( -1 ) ( 2 ) ( 3 ) ( 4 ) ).
    expected = VALUE #( ( 2 ) ( 0 ) ( 0 ) ( 1 ) ).

    result = z_humaneval_152=>compare( game = game guess = guess ).

    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

ENDCLASS.