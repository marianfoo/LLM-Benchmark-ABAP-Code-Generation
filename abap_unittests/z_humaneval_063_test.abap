CLASS ltcl_test_fibfib DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_fibfib_2 FOR TESTING,
      test_fibfib_1 FOR TESTING,
      test_fibfib_5 FOR TESTING,
      test_fibfib_8 FOR TESTING,
      test_fibfib_10 FOR TESTING,
      test_fibfib_12 FOR TESTING,
      test_fibfib_14 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_fibfib IMPLEMENTATION.
  METHOD test_fibfib_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 2 )
      exp = 1 ).
  ENDMETHOD.

  METHOD test_fibfib_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 1 )
      exp = 0 ).
  ENDMETHOD.

  METHOD test_fibfib_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 5 )
      exp = 4 ).
  ENDMETHOD.

  METHOD test_fibfib_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 8 )
      exp = 24 ).
  ENDMETHOD.

  METHOD test_fibfib_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 10 )
      exp = 81 ).
  ENDMETHOD.

  METHOD test_fibfib_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 12 )
      exp = 274 ).
  ENDMETHOD.

  METHOD test_fibfib_14.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_063=>fibfib( 14 )
      exp = 927 ).
  ENDMETHOD.
ENDCLASS.